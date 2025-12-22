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
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import build_linear_layer
from mmdet3d_plugin.core.bbox.util import normalize_bbox
import torch.utils.checkpoint as cp
from .stereo import StereoQueryGenerator


@HEADS.register_module()
class HybridQueryGenerator(BaseModule):
    def __init__(self,
                 return_cfg=dict(),
                 wich_cp=False,

                 with_avg_pool=True,
                 with_cls=False,
                 with_size=False,
                 with_center=True,
                 with_heading=False,
                 with_attr=False,
                 attr_dim=2,    # vx, vy
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=10,

                 reg_class_agnostic=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 extra_encoding=dict(
                     num_layers=2,
                     feat_channels=[512, 256],
                     features=[
                         dict(
                             type='intrinsic',
                             in_channels=16,
                         ), ]
                 ),
                 num_shared_convs=1,
                 num_shared_fcs=1,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_size_convs=0,
                 num_size_fcs=0,
                 num_center_convs=0,
                 num_center_fcs=0,
                 num_heading_convs=0,
                 num_heading_fcs=0,
                 num_attr_convs=0,
                 num_attr_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_depth_sample=8,
                 loss_cls=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs
                 ):
        super(HybridQueryGenerator, self).__init__(init_cfg=init_cfg)

        assert with_center
        self.stereo_query_generator = StereoQueryGenerator(num_depth_sample)
        self.confidence_learning = ConfidenceGate(in_channels=256, proj_dim=32, hidden=32)
        self.return_cfg = return_cfg
        self.with_cp = wich_cp

        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_size = with_size
        self.with_center = with_center
        self.with_heading = with_heading
        self.with_attr = with_attr
        self.attr_dim = attr_dim
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg

        self.loss_cls = loss_cls
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        self.debug_imgs = None

        if num_cls_convs > 0 or num_size_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_size:
            assert num_size_convs == 0 and num_size_fcs == 0
        if not self.with_heading:
            assert num_heading_convs == 0 and num_heading_fcs == 0
        if not self.with_center:
            assert num_center_convs == 0 and num_center_fcs == 0
        if not self.with_attr:
            assert num_attr_convs == 0 and num_attr_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_size_convs = num_size_convs
        self.num_size_fcs = num_size_fcs
        self.num_center_convs = num_center_convs
        self.num_center_fcs = num_center_fcs
        self.num_heading_convs = num_heading_convs
        self.num_heading_fcs = num_heading_fcs
        self.num_attr_convs = num_attr_convs
        self.num_attr_fcs = num_attr_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.extra_encoding = extra_encoding

        # build prediction layers
        last_layer_dim = self.build_shared_nn()
        self.shared_out_channels = last_layer_dim
        last_layer_dim = self.build_extra_encoding()
        self.shared_out_channels = last_layer_dim
        self.build_branch()
        self.build_predictor()

        self.relu = nn.ReLU(inplace=True)

        if init_cfg is None:
            self.init_cfg = []

            if self.with_cls:
                self.init_cfg += [dict(type='Normal', std=0.01, override=dict(name='fc_cls'))]
            if self.with_center:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_center'))]
            if self.with_size:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_size'))]
            if self.with_heading:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_heading'))]
            if self.with_attr:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_attr'))]

            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='size_fcs'),
                        dict(name='heading_fcs'),
                        dict(name='center_fcs'),
                        dict(name='attr_fcs'),
                        dict(name='extra_enc'),
                    ])
            ]

        self.fp16_enabled = False

    def build_shared_nn(self):
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        return last_layer_dim

    def build_extra_encoding(self):
        in_channels = self.shared_out_channels
        feat_channels = self.extra_encoding['feat_channels']
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels] * self.extra_encoding['num_layers']
        else:
            assert len(feat_channels) == self.extra_encoding['num_layers']

        for encoding in self.extra_encoding['features']:
            in_channels = in_channels + encoding['in_channels']

        module = []
        assert self.extra_encoding['num_layers'] > 0
        for i in range(self.extra_encoding['num_layers']):
            module.append(nn.Linear(in_channels, feat_channels[i]))
            # module.append(nn.LayerNorm(feat_channels[i]))
            module.append(nn.ReLU(inplace=True))
            in_channels = feat_channels[i]
        module = nn.Sequential(*module)
        self.extra_enc = module

        return feat_channels[-1]

    def build_predictor(self):
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_size:
            out_dim_size = (3 if self.reg_class_agnostic else 3 * self.num_classes)
            self.fc_size = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.size_last_dim,
                out_features=out_dim_size)
        if self.with_heading:
            # sin ry, cos ry
            out_dim_heading = 2
            self.fc_heading = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.heading_last_dim,
                out_features=out_dim_heading)
        if self.with_center:
            # cx, cy, d
            out_dim_center = 3
            self.fc_center = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.center_last_dim,
                out_features=out_dim_center)
        if self.with_attr:
            out_dim_attr = self.attr_dim
            self.fc_attr = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.attr_last_dim,
                out_features=out_dim_attr)

    def build_branch(self):
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        self.size_convs, self.size_fcs, self.size_last_dim = \
            self._add_conv_fc_branch(
                self.num_size_convs, self.num_size_fcs, self.shared_out_channels)

        self.heading_convs, self.heading_fcs, self.heading_last_dim = \
            self._add_conv_fc_branch(
                self.num_heading_convs, self.num_heading_fcs, self.shared_out_channels)

        self.center_convs, self.center_fcs, self.center_last_dim = \
            self._add_conv_fc_branch(
                self.num_center_convs, self.num_center_fcs, self.shared_out_channels)

        self.attr_convs, self.attr_fcs, self.attr_last_dim = \
            self._add_conv_fc_branch(
                self.num_attr_convs, self.num_attr_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_size_fcs == 0:
                self.size_last_dim *= self.roi_feat_area
            if self.num_heading_fcs == 0:
                self.size_heading_dim *= self.roi_feat_area
            if self.num_center_fcs == 0:
                self.size_center_dim *= self.roi_feat_area
            if self.num_attr_fcs == 0:
                self.size_attr_dim *= self.roi_feat_area

    @property
    def custom_cls_channels(self):
        if self.loss_cls is None:
            return False
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def get_output(self, x, convs, fcs):
        for conv in convs:
            x = conv(x)
        if x.dim() > 2:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
        for fc in fcs:
            x = self.relu(fc(x))
        return x

    @force_fp32(apply_to=('center_pred', ))
    def center2lidar(self, center_pred, intrinsic, extrinsic):
        # [z, z, 1, 1] * pts_img_home.T = intrinsic @ extrinsic.T @ pts_lidar_hom.T
        center_img = torch.cat([center_pred[:, :2] * center_pred[:, 2:3], center_pred[:, 2:3]], dim=1) # (xc, yc, zc) = (u*d, v*d, d), center_pred = (u, v, d)
        center_img_hom = torch.cat([center_img, center_img.new_ones([center_img.shape[0], 1])], dim=1)  # [num_rois, 4]
        lidar2img = torch.bmm(intrinsic, extrinsic.transpose(1, 2)) # matrix product
        img2lidar = torch.inverse(lidar2img).float()
        center_lidar = torch.bmm(img2lidar, center_img_hom[..., None])[:, :3, 0] # Eq.(2)
        return center_lidar

    def forward(self, x, prev_x, intrinsics, extrinsics, match, memory_reference_point, memory_intrinsics, memory_extrinsics, egopose_trans, extra_feats=dict()):
        if not self.with_cp:
            roi_feat, return_feats = self.get_roi_feat(x, extra_feats) # encoding RoI features with conv+pool
            center_pred_mono, return_feats = self.get_prediction(roi_feat, intrinsics, extrinsics, extra_feats, return_feats)
            center_pred_stereo = self.stereo_query_generator(
                x, prev_x, intrinsics, extrinsics, match.detach(),
                memory_reference_point, memory_intrinsics, memory_extrinsics, egopose_trans, extra_feats)
            alpha = self.confidence_learning(x, prev_x, match)
            center_pred = alpha * center_pred_stereo + (1 - alpha) * center_pred_mono
        else:
            roi_feat, return_feats = cp.checkpoint(self.get_roi_feat, x, extra_feats)
            center_pred, return_feats = cp.checkpoint(self.get_prediction, roi_feat, intrinsics, extrinsics, extra_feats, return_feats)
        return center_pred, return_feats

    def temporal_stereo(self, x, rois, prev_x, prev_rois, intrinsics, extrinsics, extra_feats,  **data):
        prev_exists = data['prev_exists'][0, -1]
        mid_frame = prev_exists.bool().flatten()[0].item()
        if not mid_frame:
            pass

    def get_roi_feat(self, x, extra_feats=dict()):
        # implicitly encode object location with Eq.(5): conv+pool+MLP

        return_feats = dict()
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # extra encoding
        enc_feat = [x]
        for enc in self.extra_encoding['features']:
            enc_feat.append(extra_feats.get(enc['type']))
        enc_feat = torch.cat(enc_feat, dim=1).clamp(min=-5e3, max=5e3)
        x = self.extra_enc(enc_feat)
        if self.return_cfg.get('enc', False):
            return_feats['enc'] = x

        return x, return_feats

    def get_prediction(self, x, intrinsics, extrinsics, extra_feats, return_feats):
        # separate branches
        x_cls = x
        x_center = x
        x_size = x
        x_heading = x
        x_attr = x

        out_dict = {}
        for output in ['cls', 'size', 'heading', 'center', 'attr']:
            out_dict[f'x_{output}'] = self.get_output(eval(f'x_{output}'), getattr(self, f'{output}_convs'),
                                                      getattr(self, f'{output}_fcs'))
            if self.return_cfg.get(output, False):
                return_feats[output] = out_dict[f'x_{output}']

        x_cls = out_dict['x_cls']
        x_center = out_dict['x_center']
        x_size = out_dict['x_size']
        x_heading = out_dict['x_heading']
        x_attr = out_dict['x_attr']

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        size_pred = self.fc_size(x_size) if self.with_size else None
        heading_pred = self.fc_heading(x_heading) if self.with_heading else None
        center_pred = self.fc_center(x_center) if self.with_center else None # linear layer: RoI features -> 2.5D coordinates (u, v, d)
        attr_pred = self.fc_attr(x_attr) if self.with_attr else None

        center_lidar = self.center2lidar(center_pred, intrinsics, extrinsics) # Eq.(2)

        return center_lidar, return_feats



class ConfidenceGate(nn.Module):
    def __init__(self, in_channels=256, proj_dim=32, use_photo=False, use_reproj=False, hidden=32):
        super().__init__()
        self.use_photo = use_photo
        self.use_reproj = use_reproj

        self.roi_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, proj_dim, bias=True),
            nn.LayerNorm(proj_dim),
        )

        in_dim = 4  # [1-p_dummy, p_max, p_gap, -entropy]
        in_dim += 1  # cos_sim
        if use_photo:
            in_dim += 1
        if use_reproj:
            in_dim += 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    # ---------- shape helpers ----------
    @staticmethod
    def _ensure_bdim_x(feat):
        if feat.dim() == 4:  # (M,C,G,G)
            return feat.unsqueeze(0)
        if feat.dim() == 5:  # (B,M,C,G,G)
            return feat
        raise ValueError(f'Unexpected feat shape: {feat.shape}')

    @staticmethod
    def _ensure_bdim_match(match):
        if match.dim() == 2:  # (M,N+1)
            return match.unsqueeze(0)
        if match.dim() == 3:  # (B,M,N+1)
            return match
        raise ValueError(f'Unexpected match shape: {match.shape}')

    @staticmethod
    def _ensure_bdim_vec(vec, B, M):
        if vec is None:
            return None
        if vec.dim() == 1:  # (M,)
            return vec.unsqueeze(0)  # (1,M)
        if vec.dim() == 2:  # (B,M)
            return vec
        raise ValueError(f'Unexpected vec shape: {vec.shape}')

    # ---------- stats from match (robust to N==0) ----------
    @torch.no_grad()
    def _stats_from_match(self, match, eps=1e-9):
        # match: (B,M,N+1)
        B, M, NP1 = match.shape
        assert NP1 >= 1, "match 必须至少包含 1 列 dummy"
        p_dummy = match[..., -1]  # (B,M)

        if NP1 == 1:
            zeros = p_dummy.new_zeros(B, M)
            return p_dummy, zeros, zeros, zeros, zeros  # p_max, p_gap, ent, real_mass
        else:
            real = match[..., :-1]          # (B,M,N)
            real_mass = real.sum(dim=-1)    # (B,M)

            N = real.shape[-1]
            if N >= 2:
                topk = real.topk(k=2, dim=-1).values  # (B,M,2)
                p_max = topk[..., 0]
                p_gap = topk[..., 0] - topk[..., 1]
            else:  # N==1
                p_max = real[..., 0]
                p_gap = p_max

            r = real.clamp_min(eps)
            ent = -(r * r.log()).sum(dim=-1)  # (B,M)
            return p_dummy, p_max, p_gap, ent, real_mass

    # ---------- top-1 index (robust to N==0) ----------
    @torch.no_grad()
    def _top1_index(self, match, eps=1e-9):
        B, M, NP1 = match.shape
        if NP1 == 1:
            top1 = match.new_zeros(B, M, dtype=torch.long)
            has_real = torch.zeros(B, M, dtype=torch.bool, device=match.device)
            return top1, has_real
        real = match[..., :-1]               # (B,M,N)
        real_mass = real.sum(dim=-1)         # (B,M)
        top1 = real.argmax(dim=-1)           # (B,M)
        has_real = (real_mass > eps)
        top1 = torch.where(has_real, top1, top1.new_zeros(top1.shape))
        return top1, has_real

    # ---------- gather prev by top-1 (robust to N==0) ----------
    @torch.no_grad()
    def _gather_prev_by_top1(self, prev_x, top1_idx):
        # prev_x: (B,N,C,G,G), top1_idx: (B,M)
        B, N, C, G, _ = prev_x.shape
        if N == 0:
            return prev_x.new_zeros(B, top1_idx.shape[1], C, G, G)
        prev_flat = prev_x.reshape(B * N, C, G, G)
        b_idx = torch.arange(B, device=prev_x.device)[:, None].expand_as(top1_idx)  # (B,M)
        global_idx = (b_idx * N + top1_idx).reshape(-1)                              # (B*M,)
        picked = prev_flat[global_idx]                                              # (B*M,C,G,G)
        return picked.view(B, -1, C, G, G)                                          # (B,M,C,G,G)

    def forward(self, x, prev_x, match, photo_sim=None, reproj_err=None):
        x = self._ensure_bdim_x(x)                  # (B,M,C,G,G)
        prev_x = self._ensure_bdim_x(prev_x)        # (B,N,C,G,G)
        match = self._ensure_bdim_match(match)      # (B,M,N+1)
        B, M, C, G, _ = x.shape
        _, N, _, _, _ = prev_x.shape

        photo_sim = self._ensure_bdim_vec(photo_sim, B, M) if self.use_photo else None
        reproj_err = self._ensure_bdim_vec(reproj_err, B, M) if self.use_reproj else None
        if self.use_photo:
            assert photo_sim is not None and photo_sim.shape == (B, M)
        if self.use_reproj:
            assert reproj_err is not None and reproj_err.shape == (B, M)

        with torch.no_grad():
            p_dummy, p_max, p_gap, ent, real_mass = self._stats_from_match(match)

        with torch.no_grad():
            top1_idx, has_real = self._top1_index(match)
            curr_vec = self.roi_proj(x.view(B * M, C, G, G)).view(B, M, -1)   # (B,M,P)
            prev_top1 = self._gather_prev_by_top1(prev_x, top1_idx)           # (B,M,C,G,G)
            prev_vec = self.roi_proj(prev_top1.view(B * M, C, G, G)).view(B, M, -1)

            curr_n = F.normalize(curr_vec, dim=-1)
            prev_n = F.normalize(prev_vec, dim=-1)
            cos_sim = (curr_n * prev_n).sum(dim=-1)                           # (B,M)
            cos_sim = torch.where(has_real, cos_sim, cos_sim.new_zeros(cos_sim.shape))

        feats = [1.0 - p_dummy, p_max, p_gap, -ent, cos_sim]
        if self.use_photo:
            feats.append(photo_sim.detach())
        if self.use_reproj:
            feats.append((-reproj_err).detach())
        feat = torch.stack(feats, dim=-1).float()                             # (B,M,F)

        logit = self.mlp(feat)                                                # (B,M,1)
        c = torch.sigmoid(logit)                                              # (B,M,1)

        no_real = (real_mass <= 1e-6).unsqueeze(-1)                           # (B,M,1)
        if no_real.any():
            c = torch.where(no_real, c.new_zeros(c.shape), c)
        c = torch.clamp(c, 1e-3, 1 - 1e-3)

        return c[0]
