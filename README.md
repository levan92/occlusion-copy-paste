# Occlusion Copy & Paste

Code for [Humans need not label more humans: Occlusion Copy & Paste for Occluded Human Instance Segmentation]()

This repository implements Occlusion Copy & Paste (OC&P) as described in our paper. Most of the repository is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Install 

- Make sure mmcv is installed (our mmcv version is 1.5.3, but any works as long as it is compatible with our mmdet version) 
- Installing this repository will install MMDetection version 2.25.0 with our custom codes for OC&P ([see details below](#core-implementation))

```
python3 -m pip install . 
```
or 

```
python3 -m pip install -e . 
```
for editable installation

## Details 

### Core implementation

Implementation is built upon [MMDetection](https://github.com/open-mmlab/mmdetection).

Main changes in: 
- [`mmdet/datasets/pipelines/transforms.py`](./mmdet/datasets/pipelines/transforms.py
  - Search for inline comments: `Code added as part of Occlusion Copy Paste`

Only the eventual Occlusion Copy & Paste implementation is provided in this repository for brevity. If you require implementation codes for other add-ons, please contact us.  

Training config files for OCP can be found in: 
- Mask-RCNN, R50 backbone, FPN:
  - Vanilla Baseline (w/o copy paste): [`configs/mask_rcnn/coco_human-vanilla_baseline-75epochs.py`](configs/mask_rcnn/coco_human-vanilla_baseline-75epochs.py)
  - Basic Copy & Paste: [`configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py`](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py)
  - Occluded Copy & Paste: [`configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py`](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py)
- Mask2Former, Swin-S backbone: 
  - Vanilla Baseline (w/o copy paste): [`configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune.py`](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune.py)
  - Occluded Copy & Paste: [`configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py`](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py)

Batch sizes in training configs are for our training hardware: 1 node with 4 x NVIDIA V100 GPUs

### (additional) Pose2Seg implementation 

- OCP is additionally implementated for training with [Pose2Seg](https://github.com/liruilong940607/Pose2Seg). 
- Please refer to [./Pose2Seg_OCP/](./Pose2Seg_OCP). 
- Pose2Seg is implemented in plain PyTorch, so our OC&P code implementation there should be more suitable for any new adaptations elsewhere that don't use MMDetection. 
- Main efforts in implementing OC&P there is additionally handling copy & pasting and augmentation of keypoints (which wasn't considered in Mask RCNN or Mask2Former).


## Datasets

### Expected structure

(modify accordingly)

.data/
├── COCO2017/
│   ├── annotations/
│   │   ├──instances_train2017.json
│   │   ├──instances_val2017.json
│   │   └──instances_val_person2017.json
│   ├── train2017/
│   └── val2017/           
└── OCHuman                    
    ├── images/ 
    ├── ochuman_coco_format_val_range_0.00_1.00.json 
    ├── ochuman_coco_format_test_range_0.00_1.00.json
    ├── [ochuman_coco_format_val_range_0.00_1.00_full_labelled.json]()
    └── [ochuman_coco_format_test_range_0.00_1.00_full_labelled.json]()

### Dataset Download

- COCO 2017 can be downloaded [here](https://cocodataset.org/#download). 
- OCHuman can be downloaded [here](https://github.com/liruilong940607/OCHumanApi).

### OCHuman Fully Labelled

OCHuman Fully Labelled (FL) is introduced in our paper for fairer evaluation, as [original OCHuman](https://github.com/liruilong940607/OCHumanApi) val & test sets contain images that are not exhaustively labelled (with ground-truth masks). We provide here a subset of the OCHuman sets that contain exhaustively labelled ones. This subset is derived from OCHuman's original json label files (`ochuman.json`), where they exhaustively label the bounding boxes for most of the human instances (of course there are ones that are missed out, but mostly negligible), but only selectively label pose keypoints & masks. This subset contains images that have bounding boxes with labelled GT masks. All credits of images and annotations go to OCHuman creators. JSON files of this subsets can be downloaded from the links in the folder tree above. 

## Train

- We train with multi-gpu [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Look into said bash script to change your paths and namings accordingly

```
./tools/dist_train.sh
```

## Evaluation

- Look into said bash script to change your paths and namings accordingly

```
./tools/dist_test.sh
```

## Results

See more details and info in paper.


|         Model                  |  OCHuman AP Val | OCHuman AP Test | OCHuman(FL) AP Val | OCHuman(FL) AP Test | Config |   Weights   |
| :----------------------------: | :-------------: | :-------------: | :----------------: | :-----------------: | :----: | :---------: |
|   Pose2Seg                     |         -       |       -         |        22.8        |        22.9         |   -    |  their repo |
|  + Occlusion C&P (ours)        |         -       |       -         |        25.3        |        25.1         |   -    |  their repo |
|   Mask R-CNN (pretrained)      |       14.9      |      14.9       |        24.5        |        24.9         |   -    |  their repo |
|   Mask R-CNN (vanilla trained) |       16.5      |      16.6       |        27.0        |        27.4         |   -    |  their repo |
|  + Occlusion C&P (ours)        |       19.5      |      18.5       |        30.7        |        29.9         |   -    |  their repo |
|   Mask2Former (pretrained)     |       25.9      |      25.4       |        43.2        |        44.7         |   -    |  their repo |
|   Mask2Former (vanilla trained)|       26.7      |      26.3       |        45.2        |        46.4         |   -    |  their repo |
|  + Simple Copy-Paste           |       28.0      |      27.7       |        48.9        |        50.2         |   -    |  their repo |
|  + Occlusion C&P (ours)        |       28.9      |      28.3       |        49.3        |        50.6         |   -    |  their repo |

