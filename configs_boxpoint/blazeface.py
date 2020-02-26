# model settings
input_size = 128
model = dict(
    type='SingleStagePointDetector',
    # pretrained = '/home/yckj1758/.cache/torch/checkpoints/vgg16_caffe-292e1171.pth',
    # pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='BlazeNet',
        input_width=input_size,
        input_height=input_size,
        num_single=5,
        num_double=6),
    # backbone=dict(
    #     type='SSDVGG',
    #     input_size=input_size,
    #     depth=16,
    #     with_last_pool=False,
    #     ceil_mode=True,
    #     out_indices=(3, 4),
    #     out_feature_indices=(22, 34),
    #     l2_norm_scale=20),
    neck=None,
    bbox_head=dict(
        type='SSDBlazeHead',
        input_size=input_size,
        in_channels=(96, 96),
        num_classes=2,
        anchor_strides=(8, 16),
        # basesize_ratio_range=(0.1, 0.9),
        anchor_scales=([0.25, 0.75],
                       [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
        anchor_ratios=([1], [1]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
# model training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.01,
    max_per_img=200)
# dataset settings
dataset_type = 'BoxPointDataset'
data_root = '/ssd/yckj1758/keypoint/data/landmarks_68_voc/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                # data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'ImageSets/Main/train.txt'
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=6000,
    warmup_ratio=1.0 / 5,
    step=[100, 150, 180])
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/box_point/blazeface'
load_from = None
resume_from = None
workflow = [('train', 1)]
