_base_=[
    '../_base_/default_runtime.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

classes = ('person',)

# model settings 
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    # the model is trained from scratch, so init_cfg is None
    backbone=dict(
        frozen_stages=-1, 
        norm_eval=False, 
        norm_cfg=norm_cfg, 
        init_cfg=None
        ),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            norm_cfg=norm_cfg, 
            num_classes=len(classes),
            ),
        mask_head=dict(
            norm_cfg=norm_cfg,
            num_classes=len(classes),
            )
        )
    )

# dataset settings

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='OccCopyPaste',
        prob=0.8,
        basket_size=5,
        paste_num=[1,10],
        min_size_paste=0.0,
        min_size_occ=0.01,
        targeted_paste_prob=0.0,
        targeted_paste_buffer=0.4,
        aug_paste_geom_jitter=False,
        aug_paste_img_jitter=False,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = 'data/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='RepeatDataset',
            times=3,
            dataset=dict(
                type='CocoDataset',
                ann_file=data_root + 'COCO2017/annotations/instances_train2017.json',
                img_prefix=data_root + 'COCO2017/train2017/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations',
                            with_bbox=True,
                            with_mask=True,
                            poly2mask=True),
                        ],
                filter_empty_gt=True,
                )
            ),
        pipeline=train_pipeline
        ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(type='CocoDataset',
                ann_file=data_root + 'COCO2017/annotations/instances_val2017.json',
                img_prefix=data_root + 'COCO2017/val2017/',
                classes=classes,
                pipeline=test_pipeline,
                ),
            dict(type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_val_range_0.00_1.00.json',
                img_prefix=data_root + 'OCHuman/images/',
                classes=classes,
                pipeline=test_pipeline,
                ),
            ],
        ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                ann_file=data_root + 'COCO2017/annotations/instances_val2017.json',
                img_prefix=data_root + 'COCO2017/val2017/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline
            ),
            dict(
                type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_val_range_0.00_1.00.json',
                img_prefix=data_root + 'OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline,
            ),
            dict(
                type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_test_range_0.00_1.00.json',
                img_prefix=data_root + 'OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline,
            ),
            dict(type='CocoDataset',
                ann_file=data_root + 'COCO2017/annotations/instances_val_person2017.json',
                img_prefix=data_root + 'COCO2017/val2017/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline,
            ),
            dict(
                type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_val_range_0.00_1.00_full_labelled.json',
                img_prefix='data/OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline,
                ),
            dict(
                type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_test_range_0.00_1.00_full_labelled.json',
                img_prefix='data/OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline
                ),
            ]
        )
    )

# optimizer from strong baseline
optimizer = dict(
    type='SGD', 
    lr=0.0125, # for bs8 (2 x 4gpus). strong baseline's LR is 0.1 for bs64
    momentum=0.9, 
    weight_decay=0.00004
    )  
optimizer_config = dict(grad_clip=None)

# from strong baseline
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.067,
    step=[22, 24]
    )
runner = dict(type='EpochBasedRunner', max_epochs=25)

evaluation = dict(
        interval=1, 
        save_best='1_segm_mAP', 
        metric=['bbox', 'segm'],
        )

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='caphuman',
        #         name='basic_copy_paste'
        #         ),
        #     out_suffix=('.log.json', '.log', '.py')
        #     )
        ]
    )
