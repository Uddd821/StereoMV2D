# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.losses import accuracy
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from mmdet3d_plugin.core.bbox.util import normalize_bbox
import torch.utils.checkpoint as cp



def _mask_out_dummy_and_safe_normalize(match, eps=1e-6, fallback='uniform'):
    had_batch = (match.dim() == 3)
    if not had_batch:
        match = match.unsqueeze(0)  # (1,M,N+1)

    real = match[..., :-1]                   # (B,M,N)
    if real.size(-1) == 0:
        w = real
        if not had_batch:
            w = w.squeeze(0)
        return w

    s = real.sum(dim=-1, keepdim=True)       # (B,M,1)
    bad = (s <= eps)
    w = real / (s + eps)

    if bad.any():
        if fallback == 'uniform':
            uni = real.new_full(real.shape, 1.0 / real.size(-1))
            w = torch.where(bad, uni, w)
        else:
            w = torch.where(bad, torch.zeros_like(w), w)

    if not had_batch:
        w = w.squeeze(0)
    return w


def _select_topk_mass(w, cap=8, mass=0.9):
    had_batch = (w.dim() == 3)
    if not had_batch:
        w = w.unsqueeze(0)

    cap = min(cap, w.size(-1))
    vals, idx = torch.topk(w, cap, dim=-1)     # (B,M,cap)
    cum = vals.cumsum(dim=-1)                  # (B,M,cap)
    ge = (cum >= mass)
    any_ge = ge.any(dim=-1)                    # (B,M)
    first_ge = ge.float().argmax(dim=-1) + 1   # (B,M) in [1..cap]
    K_dyn = torch.where(any_ge, first_ge, torch.full_like(first_ge, cap))

    if not had_batch:
        vals, idx, K_dyn = vals.squeeze(0), idx.squeeze(0), K_dyn.squeeze(0)
    return vals, idx, K_dyn


def _apply_soft_gate(vals, K_dyn, beta=0.3, eps=1e-6):
    had_batch = (vals.dim() == 3)
    if not had_batch:
        # vals: (M,cap) -> (1,M,cap)
        vals = vals.unsqueeze(0)
        K_dyn = K_dyn.unsqueeze(0)

    B, M, cap = vals.shape
    idx_last = (K_dyn - 1).clamp(min=0).unsqueeze(-1).to(dtype=torch.long)  # (B,M,1)
    kth = torch.gather(vals, -1, idx_last)  # (B,M,1)

    gate = torch.sigmoid(beta * (vals - kth))  # (B,M,cap)

    w = vals * gate
    s = w.sum(dim=-1, keepdim=True).clamp_min(eps)
    w = w / s

    if not had_batch:
        w = w.squeeze(0)
    return w


@HEADS.register_module()
class StereoQueryGenerator(BaseModule):
    def __init__(self, num_depth_sample=8, init_cfg=None, **kwargs):
        super(StereoQueryGenerator, self).__init__(init_cfg=init_cfg)

        self.num_depth_sample = num_depth_sample

        self.mlp = nn.Sequential(
            nn.Linear(num_depth_sample, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_depth_sample)
        )

    def forward(self, x, prev_x, intrinsics, extrinsics, match, memory_reference_point, memory_intrinsics, memory_extrinsics, egopose_trans, extra_feats=dict()):
        # Obtained 3D prior reference points for current detections from history queries by weighted sum
        ref_prior = self.weighted_ref_prior(match, memory_reference_point)  # (B,M,3) lidar coordinates
        B_, M_, _ = ref_prior.shape
        assert B_ == 1  # only support batch_size = 1

        # lidar -> camera -> image: instristic @ extrinsic @ ref_prior
        ones = torch.ones((B_, M_, 1), device=ref_prior.device, dtype=match.dtype)
        ref_prior_homo = torch.cat((ref_prior, ones), dim=-1) # (B,M,4)
        lidar2img = torch.bmm(intrinsics, extrinsics.transpose(1, 2)).float() # (M,4,4) lidar -> camera -> img(RoI)
        ref_roi = torch.bmm(lidar2img, ref_prior_homo.permute(1, 2, 0))[:, :3, 0] # (M, 3), (ud, vd, d), 2.5D RoI points in RoI coordinate system
        denom = ref_roi[:, 2:3].clamp_min(1e-3)
        center_roi = torch.cat([ref_roi[:, :2] / denom, ref_roi[:, 2:3]], dim=1) # (M, 3), (u, v, d), 2.5D RoI points in image coordinate system

        # SID depth sampling
        center_samp = self.sid_depth_sampling(center_roi, D=self.num_depth_sample, alpha=1.2)

        # Temporal stereo matching
        depth_prob, depth_pred = self.dense_temporal_stereo(center_samp, x, prev_x, match[0], intrinsics, extrinsics, memory_intrinsics, memory_extrinsics, egopose_trans[0])
        # depth_prob, depth_pred = self.sparse_temporal_stereo(center_samp, x, prev_x, match[0], intrinsics, extrinsics, memory_intrinsics, memory_extrinsics, egopose_trans[0])
        center_pred = torch.cat([center_roi[:, :2], depth_pred.unsqueeze(-1)], dim=1) # (u, v, d)
        center_img = torch.cat([center_pred[:, :2] * center_pred[:, 2:3], center_pred[:, 2:3]], dim=1)  # (xc, yc, zc) = (u*d, v*d, d)
        center_img_hom = torch.cat([center_img, center_img.new_ones([center_img.shape[0], 1])], dim=1)  # [num_rois, 4]
        img2lidar = torch.inverse(lidar2img)
        center_lidar = torch.bmm(img2lidar, center_img_hom[..., None])[:, :3, 0]

        return center_lidar

    def sid_depth_sampling(self, center_roi, D=8, alpha=1.2):
        """
        STS: SID depth sampling
        Args:
            center_roi: (M, 3), each row is (u, v, d) in the RoI patch
            D: number of depth samples
            alpha: range factor, sampling depth from [d/alpha, d*alpha]
        Returns:
            sampled_points: (M, D, 3), sampled (u, v, depth)
        """
        M, _ = center_roi.shape
        device = center_roi.device
        u, v, d = center_roi[:, 0], center_roi[:, 1], center_roi[:, 2]

        d_min = (d / alpha).clamp_min(1e-3)
        d_max = (d * alpha).clamp_min(1e-3)
        # i=0,...,D-1
        idx = torch.arange(D, device=device).float().view(1, D)  # (1, D)
        # SID 
        depth_samples = d_min.unsqueeze(1) * (d_max / d_min).unsqueeze(1) ** (idx / (D - 1))  # (M, D)

        u_expand = u.unsqueeze(1).expand(-1, D)  # (M, D)
        v_expand = v.unsqueeze(1).expand(-1, D)  # (M, D)
        sampled_points = torch.stack([u_expand, v_expand, depth_samples], dim=-1)  # (M, D, 3)

        return sampled_points

    def weighted_ref_prior(self, match, memory_reference_point,
                           eps=1e-6, cap=8, mass=0.9, beta=0.3, fallback='uniform'):
        had_batch = (match.dim() == 3)
        if not had_batch:
            match = match.unsqueeze(0)  # (1,M,N+1)
            memory_reference_point = memory_reference_point.unsqueeze(0)  # (1,N+1,3)

        B, M, Np1 = match.shape
        N_real = max(0, Np1 - 1)

        if N_real == 0:
            ref_prior = memory_reference_point.new_zeros(B, M, 3)
            if not had_batch:
                ref_prior = ref_prior.squeeze(0)
            return ref_prior

        s = _mask_out_dummy_and_safe_normalize(match, eps=eps, fallback=fallback)  # (B,M,N)

        if s.numel() == 0 or s.size(-1) == 0:
            ref_prior = memory_reference_point.new_zeros(B, M, 3)
            if not had_batch:
                ref_prior = ref_prior.squeeze(0)
            return ref_prior

        pts = memory_reference_point[..., :-1, :]  # (B,N,3)

        vals, idx, K_dyn = _select_topk_mass(s, cap=cap, mass=mass)  # (B,M,cap)
        w_cap = _apply_soft_gate(vals, K_dyn, beta=beta)  # (B,M,cap)

        B, M, cap_ = idx.shape
        pts_exp = pts.unsqueeze(1).expand(B, M, pts.size(-2), pts.size(-1))  # (B,M,N,3)
        idx_exp = idx.unsqueeze(-1).expand(B, M, cap_, pts.size(-1))  # (B,M,cap,3)
        pts_k = torch.gather(pts_exp, 2, idx_exp)  # (B,M,cap,3)

        ref_prior = (w_cap.unsqueeze(-1) * pts_k).sum(dim=2)  # (B,M,3)

        if not had_batch:
            ref_prior = ref_prior.squeeze(0)
        return ref_prior

    def sparse_temporal_stereo(self, ref_pts_cands, curr_x, prev_x, match,
                               intrinsics_curr, extrinsics_curr,
                               memory_intrinsics, memory_extrinsics,
                               egopose_trans, topk_cap=8, topk_mass=0.9, topk_beta=0.3,
                               sim_temp=1.0, eps=1e-6):

        device = ref_pts_cands.device

        M, D, _ = ref_pts_cands.shape
        C, G = curr_x.shape[1], curr_x.shape[2]

        s = _mask_out_dummy_and_safe_normalize(match, eps=eps)  # (M,N)

        if s.numel() == 0 or (s.sum(dim=1) <= eps).all():
            prob = torch.full((M, D), 1.0 / D, device=device, dtype=ref_pts_cands.dtype)
            z_pred = ref_pts_cands[..., 2].mean(dim=-1)
            return prob, z_pred

        vals, idx, K_dyn = _select_topk_mass(s, cap=topk_cap, mass=topk_mass)  # (M,cap), (M,cap), (M,)
        k_match = _apply_soft_gate(vals, K_dyn, beta=topk_beta)  # (M,cap)

        uv1 = ref_pts_cands[..., :2]  # (M,D,2)
        depth = ref_pts_cands[..., 2:3]  # (M,D,1)
        uv1d = torch.cat([uv1 * depth, depth, torch.ones_like(depth)], dim=-1).unsqueeze(-1)  # (M,D,4,1)

        lidar2img_cur = torch.bmm(intrinsics_curr, extrinsics_curr.transpose(1, 2)).unsqueeze(1).float()  # (M,1,4,4)
        img2lidar_cur = torch.linalg.inv(lidar2img_cur)  # (M,1,4,4)
        pts3d_cur = (img2lidar_cur @ uv1d).squeeze(-1)  # (M,D,4)

        prev_intrinsics_real = memory_intrinsics[:-1]  # (N,4,4)
        prev_extrinsics_real = memory_extrinsics[:-1]  # (N,4,4)
        prev_feat_real = prev_x[:-1]  # (N,C,G,G)
        cur2prev = torch.linalg.inv(egopose_trans) 
        cur2prev = cur2prev[:-1] if cur2prev.size(0) == prev_intrinsics_real.size(0) + 1 else cur2prev  

        curr2prev_select = cur2prev[idx]  # (M,cap,4,4)
        prev_intrinsics_select = prev_intrinsics_real[idx]  # (M,cap,4,4)
        prev_extrinsics_select = prev_extrinsics_real[idx]  # (M,cap,4,4)
        prev_feat_select = prev_feat_real[idx]  # (M,cap,C,G,G)

        pts3d_cur = pts3d_cur[:, None, :, :, None]  # (M,1,D,4,1)
        pts3d_prev = (curr2prev_select[:, :, None, :, :] @ pts3d_cur).squeeze(-1)  # (M,cap,D,4)

        lidar2img_prev = torch.bmm(
            prev_intrinsics_select.reshape(-1, 4, 4),
            prev_extrinsics_select.reshape(-1, 4, 4).transpose(1, 2)
        ).float().reshape(M, -1, 4, 4)  # (M,cap,4,4)

        proj = (lidar2img_prev[:, :, None, :, :] @ pts3d_prev.unsqueeze(-1)).squeeze(-1)  # (M,cap,D,4)
        denom = proj[..., 2:3].clamp_min(1e-3)
        uvd_prev = torch.cat([proj[..., :2] / denom, proj[..., 2:3]], dim=-1)  # (M,cap,D,3)

        grid = uvd_prev[..., :2]  # (M,cap,D,2)
        grid_norm = (grid / (G - 1)) * 2 - 1
        grid_norm = grid_norm.clamp(-1.05, 1.05)  
        grid_norm = grid_norm.unsqueeze(2)  # (M,cap,D,1,2)

        prev_feat_select = prev_feat_select.reshape(M * k_match.size(1), C, G, G).detach()
        warped = F.grid_sample(
            prev_feat_select, grid_norm.reshape(M * k_match.size(1), D, 1, 2),
            align_corners=True
        )  # (M*cap, C, D, 1)
        warped = warped.view(M, k_match.size(1), C, D)  # (M,cap,C,D)

        curr_feat = F.adaptive_avg_pool2d(curr_x, 1).squeeze(-1).squeeze(-1)  # (M, C)
        curr_feat = curr_feat[:, None, :, None]  # (M,1,C,1)
        sim = (warped * curr_feat).sum(dim=2) / (C ** 0.5)  # (M,cap,D)

        sim_weighted = (sim * k_match.unsqueeze(-1)).sum(dim=1)  # (M, D)
        sim_refined = self.mlp(sim_weighted)

        prob = F.softmax(sim_refined / max(sim_temp, 1.5), dim=-1)  # (M, D)
        z_pred = (prob * ref_pts_cands[..., 2]).sum(dim=-1)  # (M,)
        return prob, z_pred

    def dense_temporal_stereo(self, ref_pts_cands, curr_x, prev_x, match,
                              intrinsics_curr, extrinsics_curr,
                              memory_intrinsics, memory_extrinsics, egopose_trans,
                              topk_cap=8, topk_mass=0.9, topk_beta=0.3,
                              sim_temp=1.0, eps=1e-6):

        device = ref_pts_cands.device
        dtype = ref_pts_cands.dtype

        M, D, _ = ref_pts_cands.shape
        C, G = curr_x.shape[1], curr_x.shape[2]

        s = _mask_out_dummy_and_safe_normalize(match, eps=eps)  # (M,N)
        if s.numel() == 0 or (s.sum(dim=1) <= eps).all():
            prob = torch.full((M, D), 1.0 / D, device=device, dtype=dtype)
            z_pred = ref_pts_cands[..., 2].mean(dim=-1)
            return prob, z_pred

        vals, idx, K_dyn = _select_topk_mass(s, cap=topk_cap, mass=topk_mass)  # (M,cap)
        k_match = _apply_soft_gate(vals, K_dyn, beta=topk_beta)  # (M,cap)
        cap = k_match.size(1)

        ys = torch.arange(G, device=device, dtype=dtype)
        xs = torch.arange(G, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (G, G)
        uv_grid = torch.stack([xx, yy], dim=-1)  # (G, G, 2)
        uv_grid = uv_grid.unsqueeze(0).unsqueeze(0).expand(M, D, G, G, 2)  # (M, D, G, G, 2)

        depth = ref_pts_cands[..., 2]  # (M, D)
        depth = depth.view(M, D, 1, 1, 1).to(dtype=dtype)  # (M, D, 1, 1, 1)
        depth = depth.expand(M, D, G, G, 1)  # (M, D, G, G, 1)

        ones = torch.ones_like(depth)  # (M, D, G, G, 1)
        uv1d = torch.cat([uv_grid * depth, depth, ones], dim=-1)  # (M, D, G, G, 4)
        uv1d = uv1d.unsqueeze(-1)  # (M, D, G, G, 4, 1)

        # img2lidar_cur: (M,1,4,4)
        lidar2img_cur = torch.bmm(intrinsics_curr, extrinsics_curr.transpose(1, 2)).unsqueeze(1).float()
        img2lidar_cur = torch.linalg.inv(lidar2img_cur)  # (M,1,4,4)

        DG2 = D * G * G
        uv1d_flat = uv1d.reshape(M, DG2, 4, 1)  # (M, DG2, 4, 1)
        pts3d_cur = (img2lidar_cur @ uv1d_flat).squeeze(-1).reshape(M, D, G, G, 4)  # (M,D,G,G,4)

        prev_intrinsics_real = memory_intrinsics[:-1]  # (N,4,4)
        prev_extrinsics_real = memory_extrinsics[:-1]  # (N,4,4)
        prev_feat_real = prev_x[:-1]  # (N,C,G,G)

        cur2prev_all = torch.linalg.inv(egopose_trans)
        if cur2prev_all.size(0) == prev_intrinsics_real.size(0) + 1:
            cur2prev_all = cur2prev_all[:-1]

        curr2prev_select = cur2prev_all[idx]  # (M,cap,4,4)
        prev_intrinsics_select = prev_intrinsics_real[idx]  # (M,cap,4,4)
        prev_extrinsics_select = prev_extrinsics_real[idx]  # (M,cap,4,4)
        prev_feat_select = prev_feat_real[idx].detach()  # (M,cap,C,G,G)

        pts3d_cur_flat = pts3d_cur.reshape(M, 1, DG2, 4, 1)  # (M,1,DG2,4,1)
        pts3d_prev = (curr2prev_select.unsqueeze(2) @ pts3d_cur_flat).squeeze(-1)  # (M,cap,DG2,4)
        pts3d_prev = pts3d_prev.reshape(M, cap, D, G, G, 4)

        lidar2img_prev = torch.bmm(
            prev_intrinsics_select.reshape(-1, 4, 4),
            prev_extrinsics_select.reshape(-1, 4, 4).transpose(1, 2)
        ).float().reshape(M, cap, 4, 4)  # (M,cap,4,4)

        proj = (lidar2img_prev.unsqueeze(2).unsqueeze(2).unsqueeze(2) @  # (M,cap,1,1,1,4,4)
                pts3d_prev.unsqueeze(-1)).squeeze(-1)  # (M,cap,D,G,G,4)

        denom = proj[..., 2:3].clamp_min(1e-3)
        uvd_prev = torch.cat([proj[..., :2] / denom, proj[..., 2:3]], dim=-1)  # (M,cap,D,G,G,3)

        grid = uvd_prev[..., :2] 
        grid_norm = (grid / (G - 1)) * 2 - 1  # -> [-1,1]
        grid_norm = grid_norm.clamp(-1.05, 1.05)
        grid_norm = grid_norm.reshape(M * cap, D * G, G, 2)  

        prev_feat_flat = prev_feat_select.reshape(M * cap, C, G, G)
        warped = F.grid_sample(prev_feat_flat, grid_norm, align_corners=True)  # (M*cap, C, D*G, G)
        warped = warped.reshape(M, cap, C, D, G, G)  # (M,cap,C,D,G,G)

        curr_feat = curr_x.unsqueeze(1).unsqueeze(2)  # (M,1,1,C,G,G)
        curr_feat = curr_feat.expand(M, cap, D, C, G, G)  # (M,cap,D,C,G,G)
        sim = (warped.permute(0, 1, 3, 2, 4, 5) * curr_feat).sum(dim=3) / (C ** 0.5)  # (M,cap,D,G,G)
        sim_mean = sim.mean(dim=(-2, -1))  # (M,cap,D)

        k_match_norm = k_match / (k_match.sum(dim=1, keepdim=True) + 1e-6)  # (M,cap)
        sim_weighted = (sim_mean * k_match_norm.unsqueeze(-1)).sum(dim=1)  # (M,D)
        sim_refined = self.mlp(sim_weighted)  # (M,D)

        prob = F.softmax(sim_refined / max(sim_temp, 1.5), dim=-1)  # (M,D)
        z_pred = (prob * ref_pts_cands[..., 2]).sum(dim=-1)  # (M,)
        return prob, z_pred
