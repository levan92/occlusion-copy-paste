classes=('person',)

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)

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
    test=dict(
        _delete_=True,
        type=dataset_type,
        ann_file=data_root + 'COCO2017/annotations/instances_val_person2017.json',
        img_prefix=data_root + 'COCO2017/val2017/',
        classes=classes,
        test_mode=True,
        pipeline=test_pipeline
    )
)

