
import json
import cv2
import random 
import numpy as np 

from mmdet.datasets import CocoDataset
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations, DefaultFormatBundle, Collect

coco_annot = '/Users/levan/Data/COCO/annotations/instances_val2017.json'
coco_dir = '/Users/levan/Data/COCO/val2017'

coco = CocoDataset(
                    ann_file = coco_annot,
                    pipeline = [
                        LoadImageFromFile(to_float32=True),
                        LoadAnnotations(with_bbox=True, with_mask=True,),
                        DefaultFormatBundle(img_to_float=True),
                        Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape'))
                    ],
                    classes=('person',),
                    data_root=coco_dir,
                )

random.seed(888)
rand_idx = random.choice(range(len(coco)))

res = coco[rand_idx]

masks = res['gt_masks'].data.masks
print(masks.shape)

kernel = np.ones((3,3),np.uint8)


for i, mask in enumerate(masks): 
    cv2.imwrite(f'mask{i}_og.jpg', mask*255)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(f'mask{i}_opened.jpg', opening*255)
