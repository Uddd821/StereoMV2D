# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from .stream_mv2d_head_base import SteamMV2DHeadBase
from mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmdet3d_plugin.models.utils.pe import pos2posemb3d, nerf_positional_encoding, pos2posemb1d
from scipy.optimize import linear_sum_assignment
import math



@HEADS.register_module()
class StreamMV2DHead(SteamMV2DHeadBase):
    def __init__(self,
                 num_frame_losses=1,
                 embed_dim=256,
                 # denoise setting
                 use_denoise=False,
                 neg_bbox_loss=False,
                 denoise_scalar=10,
                 denoise_noise_scale=1.0,
                 denoise_noise_trans=0.0,
                 denoise_weight=1.0,
                 denoise_split=0.75,
                 **kwargs):
        super(StreamMV2DHead, self).__init__(**kwargs)
        self.use_denoise = use_denoise
        self.neg_bbox_loss = neg_bbox_loss
        self.denoise_scalar = denoise_scalar
        self.denoise_noise_scale = denoise_noise_scale
        self.denoise_noise_trans = denoise_noise_trans
        self.denoise_weight = denoise_weight
        self.denoise_split = denoise_split

        self.num_frame_losses = num_frame_losses
        self.embed_dim = embed_dim
        self.reset_memory()

        self.query_embedding = nn.Sequential(
            nn.Linear(embed_dim*3//2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ego_pose_pe = MLN(180)
        self.ego_pose_queries = MLN(180)
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.roifeats_encoding = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.bboxfeats_encoding = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)
        )

        self.register_buffer('_global_iter', torch.zeros((), dtype=torch.long))

        self.sinkhorn_tau_cfg = dict(
            tau_hi=0.9,
            tau_lo=0.25, 
            warm_iters=2000, 
            decay_iters=8000, 
            mode='cos'
        )

    def prepare_for_dn(self, batch_size, reference_points, img_metas, ref_num, eps=1e-4):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d'].gravity_center, img_meta['gt_bboxes_3d'].tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d'] for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.denoise_scalar, 1).view(-1)
            known_labels = labels.repeat(self.denoise_scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.denoise_scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.denoise_scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.denoise_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.denoise_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.denoise_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0 + eps, max=1.0 - eps)
                mask = torch.norm(rand_prob, 2, 1) > self.denoise_split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.denoise_scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(self.denoise_scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + ref_num
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.denoise_scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.denoise_scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def reset_memory(self):
        # self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.memory_bbox_feats = None
        self.memory_rois = None
        self.memory_intrinsics = None
        self.memory_extrinsics = None

    def get_motion_aware_rois(self, pc_range):
        # encode the ref points
        temp_ref_points = (self.memory_reference_point - pc_range[:3]) / (pc_range[3:6] - pc_range[0:3])  # 归一化
        temp_pos = self.query_embedding(pos2posemb3d(temp_ref_points))
        # pass the encoded ref points together with the vel, timestamp and ego pose into the MLN to get the tmp_pos
        tmp_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
        tmp_ego_motion = nerf_positional_encoding(tmp_ego_motion)  # encoded motion vector
        # pass the encoded ref points together with the vel, timestamp and ego pose into the MLN to get the tmp_pos
        temp_pos = self.ego_pose_pe(temp_pos, tmp_ego_motion)  # conditional layer normalization Q_p
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())
        temp_embedding = self.ego_pose_queries(self.memory_embedding, tmp_ego_motion)  # conditional layer normalization Q_c

        motion_aware_rois = temp_embedding + temp_pos
        return motion_aware_rois

    def sinkhorn_matching_with_dummy(self, sim_matrix, prev_exists, n_iters=5, tau=0.25,
                                     col_eps=1e-6, row_eps=1e-6, mix_alpha=0.05):
        B, M, N = sim_matrix.shape
        if not prev_exists:
            P = sim_matrix.new_zeros(B, M, N)
            P[..., -1] = 1.0
            return P

        log_alpha = sim_matrix / tau
        P = torch.softmax(log_alpha, dim=-1) 

        for _ in range(n_iters):
            P = P / (P.sum(dim=-1, keepdim=True) + row_eps)

            P_real, P_dummy = P[..., :-1], P[..., -1:]
            col_sum = P_real.sum(dim=-2, keepdim=True) + col_eps

            P_real = P_real / col_sum
            if mix_alpha > 0:
                P_real = (1 - mix_alpha) * P_real + mix_alpha * (1.0 / max(1, P_real.size(-1)))

            P = torch.cat([P_real, P_dummy], dim=-1)

            P = P / (P.sum(dim=-1, keepdim=True) + row_eps)

        return P

    def _tau_schedule(self, step, tau_hi=0.9, tau_lo=0.25,
                      warm_iters=2000, decay_iters=8000, mode='cos'):
        if step < warm_iters:
            return float(tau_hi)
        t = step - warm_iters
        if t < decay_iters:
            p = t / float(decay_iters)  # 0→1
            if mode == 'cos':
                return float(tau_lo + 0.5 * (tau_hi - tau_lo) * (1 + math.cos(math.pi * p)))
            else:
                return float(tau_hi * (tau_lo / tau_hi) ** p)
        return float(tau_lo)

    def match_rois(self, curr_rois_embedding, prev_rois_embedding, prev_exists):
        curr_rois_embedding = self.input_proj(curr_rois_embedding)
        prev_rois_embedding = self.input_proj(prev_rois_embedding)
        B, M, C = curr_rois_embedding.shape
        _, N, _ = prev_rois_embedding.shape
        sim = torch.einsum('bmc,bnc->bmn', curr_rois_embedding, prev_rois_embedding) * (C ** -0.5)

        if self.training:
            tau = self._tau_schedule(int(self._global_iter.item()), **self.sinkhorn_tau_cfg)
            self._global_iter.add_(1)
        else:
            tau = 0.25

        P_soft = self.sinkhorn_matching_with_dummy(sim.detach(), prev_exists, tau=tau, n_iters=5)
        return P_soft

    def _bbox_forward_denoise(self, x, proposal_list, img_metas, **data):
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]

        rois = bbox2roi(proposal_list) # (N, 5)

        # equivalent camera intrinsic matrix
        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois) # RoI features

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)
        bbox_feats, pe = bbox_feats.split([c // 2, c // 2], dim=1)
        rois_feats = bbox_feats + pe # RoI features with 3D PE (N, 256, 7, 7)
        curr_rois_embedding = F.adaptive_avg_pool2d(self.bboxfeats_encoding(bbox_feats), 1).flatten(1).unsqueeze(0)  # (1, N, 256)
        curr_rois_embedding = F.layer_norm(curr_rois_embedding, (curr_rois_embedding.size(-1),))

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        prev_exits = data['prev_exists'][:, -self.num_frame_losses:].squeeze(1)
        B = prev_exits.size(0)
        if prev_exits == 0:
            # zero init the memory bank
            device = prev_exits.device
            fdtype = prev_exits.dtype
            eye4 = torch.eye(4, dtype=fdtype, device=device)
            self.memory_reference_point = torch.zeros(B, 1, 3, dtype=fdtype, device=device).detach()  # (B,1,3)
            self.memory_timestamp = torch.zeros(B, 1, 1, dtype=torch.float64, device=device)  # (B,1,1) 
            self.memory_egopose = eye4.view(1, 1, 4, 4).repeat(B, 1, 1, 1).contiguous()  # (B,1,4,4) 
            self.memory_velo = torch.zeros(B, 1, 2, dtype=fdtype, device=device).detach()  # (B,1,2)
            self.memory_embedding = torch.zeros(B, 1, self.embed_dim, dtype=fdtype, device=device).detach()  # (B,1,C)
            self.memory_bbox_feats = bbox_feats.new_zeros(  # (1,C,H,W)
                1, bbox_feats.size(1), bbox_feats.size(2), bbox_feats.size(3)
            ).detach()
            eye44 = torch.eye(4, device=intrinsics.device, dtype=intrinsics.dtype)
            self.memory_intrinsics = eye44.unsqueeze(0).detach()  # (1, 4, 4)
            self.memory_extrinsics = eye44.unsqueeze(0).detach()  # (1, 4, 4)
        else:
            # pre update the memory bank
            self.memory_timestamp += data['timestamp'][:,-self.num_frame_losses:].unsqueeze(-1) # delta t
            self.memory_timestamp = torch.cat([self.memory_timestamp, prev_exits.new_zeros(B, 1, 1, dtype=torch.float64)], dim=1)
            self.memory_egopose = data['ego_pose_inv'][:,-self.num_frame_losses:] @ self.memory_egopose # E(t-1->t)
            self.memory_egopose = torch.cat([self.memory_egopose, torch.tensor(np.eye(4), dtype=self.memory_egopose.dtype, device=self.memory_egopose.device)[None, None, ...]], dim=1)
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'][:,-self.num_frame_losses:].squeeze(1),
                                                                     reverse=False)

            dummy_ref = torch.zeros(B, 1, 3, device=self.memory_reference_point.device)
            self.memory_reference_point = torch.cat([self.memory_reference_point, dummy_ref], dim=1).detach()
            self.memory_embedding = torch.cat([self.memory_embedding, bbox_feats.new_zeros(B, 1, bbox_feats.size(1))], dim=1).detach()
            self.memory_velo = torch.cat([self.memory_velo, prev_exits.new_zeros(B, 1, 2)], dim=1).detach()

        pc_range = nn.Parameter(torch.tensor(self.pc_range, device=bbox_feats.device), requires_grad=False)
        # prev_rois_embedding = self.get_motion_aware_rois(pc_range) # (1, N + 1, 256)
        prev_rois_embedding = F.layer_norm(self.memory_embedding, (self.memory_embedding.size(-1),))

        match = self.match_rois(curr_rois_embedding, prev_rois_embedding, prev_exits)

        # query generator
        reference_points, return_feats = self.query_generator(bbox_feats, self.memory_bbox_feats, intrinsics, extrinsics, match,
                                                              self.memory_reference_point, self.memory_intrinsics, self.memory_extrinsics,
                                                              self.memory_egopose, extra_feats)
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        reference_points.clamp(min=0, max=1)

        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

        if self.use_denoise and self.training:
            # bbox_feats: [num_rois, c, h, w]
            n_rois, c, h, w = bbox_feats.shape
            cross_attn_mask = bbox_feats.new_ones((n_rois, n_rois + 1)).bool()
            corr[~mask] = n_rois  # [num_rois, max_corr]
            cross_attn_mask = torch.scatter(cross_attn_mask, 1, corr, 0)
            cross_attn_mask = cross_attn_mask[:, :n_rois, None, None].expand(n_rois, n_rois, h, w)

            reference_points_ori = reference_points
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(1, reference_points, img_metas[0:1],
                                                                         len(reference_points))
            reference_points = reference_points[0]
            cross_attn_mask_pad = cross_attn_mask.new_zeros(
                (len(reference_points) - len(reference_points_ori), n_rois, h, w))
            cross_attn_mask = torch.cat([cross_attn_mask_pad, cross_attn_mask])

            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                            bbox_feats[None],
                                                            torch.zeros_like(bbox_feats[None, :, 0]).bool(),
                                                            pe[None],
                                                            attn_mask=attn_mask,
                                                            cross_attn_mask=cross_attn_mask,
                                                            force_fp32=self.force_fp32, )
        else:
            mask_dict = None

            corr_feats = bbox_feats[corr]  # [num_rois, num_corrs, c, h, w]
            corr_pe = pe[corr]
            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
                                                            force_fp32=self.force_fp32, )

        # post update the memory bank
        self.memory_egopose = prev_exits.new_zeros(B, rois.size(0), 4, 4)
        self.memory_egopose = self.memory_egopose + torch.tensor(1, device=data['ego_pose'].device, dtype=data['ego_pose'].dtype).view(B, 1, 1, 1) * torch.eye(4, device=prev_exits.device)
        self.memory_timestamp = prev_exits.new_zeros(B, rois.size(0), 1, dtype=torch.float64)
        self.memory_embedding = F.adaptive_avg_pool2d(self.bboxfeats_encoding(bbox_feats), 1).flatten(1).unsqueeze(0)  # (1, N, 256)

        memo_bbox_preds = all_bbox_preds.permute(0, 2, 1, 3)
        self.memory_timestamp -= data['timestamp'][:,-self.num_frame_losses:].unsqueeze(-1)
        self.memory_egopose = data['ego_pose'][:,-self.num_frame_losses:] @ self.memory_egopose # E(t-1)
        self.memory_velo = memo_bbox_preds[..., -2:][-1]
        self.memory_reference_point = memo_bbox_preds[..., :3][-1]
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'][:,-self.num_frame_losses:].squeeze(1),
                                                                 reverse=False) 

        self.memory_bbox_feats = torch.cat([bbox_feats, bbox_feats.new_zeros(1, bbox_feats.size(1), bbox_feats.size(2), bbox_feats.size(3))], dim=0).detach()
        # self.memory_rois = rois
        eye = torch.tensor(np.eye(4), dtype=intrinsics.dtype, device=intrinsics.device)
        self.memory_intrinsics = torch.cat([intrinsics, eye.unsqueeze(0)], dim=0)
        self.memory_extrinsics = torch.cat([extrinsics, eye.unsqueeze(0)], dim=0)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))

        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        )

        return bbox_results

    def _bbox_forward(self, x, proposal_list, img_metas, **data):
        bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas, **data)
        return bbox_results

    def prepare_for_dn_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def forward_train(self,
                      x,
                      cur_img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ori_gt_bboxes_3d,
                      ori_gt_labels_3d,
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **data):
        assert len(cur_img_metas) // cur_img_metas[0]['num_views'] == 1

        num_imgs = len(cur_img_metas)

        proposal_boxes= []
        proposal_scores= []
        proposal_classes= []
        for i in range(num_imgs):
            proposal_boxes.append(proposal_list[i][:, :6])
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # position encoding
        pos_enc = self.position_encoding(x, cur_img_metas) # 3D PE (6, 256, 32, 88)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)] # (6, 512, 32, 88)

        losses = dict()

        if self.use_denoise:
            cur_img_metas[0]['gt_bboxes_3d'] = ori_gt_bboxes_3d[0]
            cur_img_metas[0]['gt_labels_3d'] = ori_gt_labels_3d[0]

        results_from_last = self._bbox_forward_train(x, proposal_boxes, cur_img_metas, **data)
        preds = results_from_last['pred']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)

        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        if results_from_last.get('dn_mask_dict', None) is not None:
            dn_mask_dict = results_from_last['dn_mask_dict']
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_dn_loss(
                dn_mask_dict)
            for i in range(len(output_known_class)):
                dn_loss_cls, dn_loss_bbox = self.bbox_head.dn_loss_single(
                    output_known_class[i], output_known_coord[i], known_bboxs, known_labels, num_tgt,
                    self.pc_range, self.denoise_split, neg_bbox_loss=self.neg_bbox_loss
                )
                losses[f'l{i}.dn_loss_cls'] = dn_loss_cls * self.denoise_weight * loss_weights[i]
                losses[f'l{i}.dn_loss_bbox'] = dn_loss_bbox * self.denoise_weight * loss_weights[i]

        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses
