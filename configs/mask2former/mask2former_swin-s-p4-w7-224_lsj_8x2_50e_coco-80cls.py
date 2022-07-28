_base_ = ['./mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py']

# classes = ('person',)
# num_things_classes = len(classes)
num_things_classes = 80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO2017/annotations/instances_train2017.json',
        img_prefix=data_root + 'COCO2017/train2017/',
        pipeline=train_pipeline,
        ),
    val=dict(
        type='ConcatDataset',
        datasets=[ 
            dict(
                type=dataset_type,
                ann_file=data_root + 'COCO2017/annotations/instances_val2017.json',
                img_prefix=data_root + 'COCO2017/val2017/',
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                ann_file=data_root + 'OCHuman/ochuman_coco_format_val_range_0.00_1.00.json',
                img_prefix=data_root + 'OCHuman/images/',
                pipeline=test_pipeline,
            ),
        ]
        ),
    test=dict(
        type='ConcatDataset',
        datasets=[ 
            dict(
                type=dataset_type,
                ann_file=data_root + 'COCO2017/annotations/instances_val2017.json',
                img_prefix=data_root + 'COCO2017/val2017/',
                pipeline=test_pipeline
            ),
            dict(
                type=dataset_type,
                ann_file=data_root + 'COCO2017/annotations/ochuman_coco_format_test_range_0.00_1.00.json',
                img_prefix=data_root + 'OCHuman/images/',
                pipeline=test_pipeline,
            )
        ]
    )
)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='improved-instance-segm',
                name='m2f_swins_person'
                ),
            out_suffix=('.log.json', '.log', '.py','.pt','.pth','.pkl')
            )
        ]
    )
