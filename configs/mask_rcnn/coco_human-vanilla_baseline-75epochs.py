_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

classes = ('person',)

# model settings 

model = dict(roi_head=dict(
    bbox_head=dict(num_classes=len(classes)),
    mask_head=dict(num_classes=len(classes))
    ))

# dataset settings

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
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

data_root = './data/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            ann_file=data_root + 'COCO2017/annotations/instances_train2017.json',
            img_prefix=data_root + 'COCO2017/train2017/',
            classes=classes,
        ),
        classes=classes,
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
                ann_file=data_root + 'OCHuman/ochuman_coco_format_val_range_0.00_1.00_full_labelled.json',
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
                img_prefix=data_root + 'OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline,
                ),
            dict(
                type='CocoDataset',
                ann_file=data_root + 'OCHuman/ochuman_coco_format_test_range_0.00_1.00_full_labelled.json',
                img_prefix=data_root + 'OCHuman/images/',
                classes=classes,
                test_mode=True,
                pipeline=test_pipeline
                ),
            ]
    ),
)

# optimizer from strong baseline
optimizer = dict( lr=0.0125 ) # for bs8 (2 x 4gpus). strong baseline's LR is 0.1 for bs64
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.067,
    step=[22, 24]
    )
runner = dict(type='EpochBasedRunner', max_epochs=25)
# 25 epochs, but 3x repeat dataset, equivalent to 75 epochs

evaluation = dict(
        interval=1, 
        save_best='1_segm_mAP', 
        metric=['bbox', 'segm'],
        )

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='ocp',
        #         name='coco_human-75eps-baseline'
        #         ),
        #     out_suffix=('.log.json', '.log', '.py')
        #     )
        ]
    )

