_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from="/datadisk/levan/mmdet/weights/fasterrcnn/faster_rcnn_r50_caffe_c4_1x_coco_20220316_150152-3f885b85.pth"

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = 'data/'

data = dict(
    train=dict(
        ann_file=data_root + 'COCO2017/annotations/instances_train2017.json',
        img_prefix=data_root + 'COCO2017/train2017/',
        pipeline=train_pipeline
        ),
    val=dict(
        ann_file=data_root + 'COCO2017/annotations/instances_val2017_mini_mini.json',
        img_prefix=data_root + 'COCO2017/val2017/',
        pipeline=test_pipeline
        
        ),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

max_iters = 100 
runner = dict(
    _delete_=True, 
    type='IterBasedRunner', 
    max_iters=max_iters
)

lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10
    )

interval = 10
workflow = [('train', interval),('val',1)]
checkpoint_config = dict(
    by_epoch=False, interval=interval)

evaluation = dict(
    interval=interval,
    metric=['bbox'])

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                project='train-tests',
                name='short-mmdetwandb'
                ),
            out_suffix=('.log.json', '.log', '.py'),
            interval=50,
            eval_interval=interval,
            by_epoch=False,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=2,
            bbox_score_thr=0.3,
            ),
        ]
    )
