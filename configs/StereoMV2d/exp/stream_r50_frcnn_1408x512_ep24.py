from configs.mv2d.data.stream_frame import num_frame_losses

_base_ = [
    '../data/stream_frame.py', '../detectors/maskrcnn_r50.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
post_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
roi_size = 7
roi_srides = [16]

model = dict(
    type='StreamMV2D',
    num_frame_losses=num_frame_losses,
    use_grid_mask=dict(
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio_range=(0.4, 0.6),
        mode=1,
        prob=0.7,
        interv_ratio=0.8
    ),
    # NOTE:
    # the FPN in faster r-cnn starts from p2
    # we use p4 (downsample rate: 16)
    base_detector=dict(
        backbone=dict(
            with_cp=False,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256, 256],
        out_channels=256,
        start_level=2,
        end_level=2,
        num_outs=1,
    ),
    roi_head=dict(
        type='StreamMV2DHead',
        num_frame_losses=num_frame_losses,
        embed_dim=256,
        pc_range=point_cloud_range,
        force_fp32=True,
        use_denoise=False, #

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),
            featmap_strides=roi_srides,
            out_channels=512, ),
        bbox_head=dict(
            type='CrossAttentionBoxHead',
            num_classes=10,
            pc_range=point_cloud_range,
            transformer=dict(
                type='MV2DTransformer',
                decoder=dict(
                    type='PETRTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='PETRTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='FlattenMHSelfAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='PETRMultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        with_cp=False,  ###use checkpoint to save memory
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=post_range,
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10),
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        ),
        query_generator=dict(
            with_avg_pool=True,
            num_shared_convs=1,
            num_shared_fcs=1,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=roi_size,
            extra_encoding=dict(
                num_layers=2,
                feat_channels=[512, 256],
                features=[dict(type='intrinsic', in_channels=16,)]
            ),
        ),
        pe=dict(
            positional_encoding=dict(
                type='SinePositionalEncoding3D', num_feats=128, normalize=True),
            strides=roi_srides,
            position_range=post_range,
            depth_num=64,
            with_fpe=True,
        ),
        box_correlation=dict(
            correlation_mode='topk_matched:1:0.0:0.0',
        ),
    ),
    train_cfg=dict(
        complement_2d_gt=0.4,
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),
        rcnn=dict(
            stage_loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range),
            sampler_cfg=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(nms_thr=1.0, use_rotate_nms=True, ),
            max_per_scene=300,
        ))
)

data = dict(
    workers_per_gpu=4,
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),   # 保守稳妥；若仍抖可试 (0.9, 0.95)
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'base_detector.backbone': dict(lr_mult=0.25),
            'roi_head.hybrid_query_generator': dict(lr_mult=0.5),
            'roi_head.hybrid_query_generator.confidence_learning': dict(lr_mult=0.5),
            'roi_head.stereo_query_generator': dict(lr_mult=0.5),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    )
)

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=20, norm_type=2)
)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,            # ←← 关键：按 iter 衰减
    warmup='linear',
    warmup_iters=5000,         # 从 500 -> 1500，更平滑进入
    warmup_ratio=1.0/3,
    min_lr_ratio=1e-3          # 最低 lr=2e-4 * 1e-3 = 2e-7
)

num_epochs = 24
num_gpus = 1
batch_size = 1
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
# runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)
# evaluation = dict(interval=2, )
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch * 3, by_epoch=False)
find_unused_parameters = False
log_config = dict(interval=50,)
checkpoint_config = dict(interval=num_iters_per_epoch * 3, by_epoch=False)

resume_from = 'work_dirs/stream_r50_frcnn_1408x512_ep24/latest.pth'
