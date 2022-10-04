# Occlusion Copy & Paste

Code for [Humans need not label more humans: Occlusion Copy & Paste for Occluded Human Instance Segmentation]()

This repository implements Occlusion Copy & Paste (OC&P) as described in our paper. Most of the repository is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Install 

- Make sure [mmcv](https://github.com/open-mmlab/mmcv) v1+ is installed, full version.
  - We usually clone `mmcv` and enter it to install with `MMCV_WITH_OPS=1 python3 -m pip install .`  (our mmcv version is 1.5.3, but any works as long as it is compatible with our mmdet version) 
- Adhere to mmdet/mmcv's PyTorch/CUDA requirements (our PyTorch version is 1.9.0 and CUDA version is 11.1, any works as long as it is compatible to mmdet/mmcv)
- Installing this repository will install MMDetection with our custom codes for OC&P ([see details below](#core-implementation))
  - Code here is based of mmdet v2+ (current master branch of mmdet) 
    - Mmdet v3 is available now and we have separate plans to upgrade to v3. Not supported in this repo. 

```
cd occlusion-copy-paste/
python3 -m pip install . 
```
or 

```
python3 -m pip install -e . 
```
for editable installation

## Details 

### Core implementation

- Implementation is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) version 2+
- Occlusion Copy & Paste logic is contained within [`mmdet/custom/ocp.py`](./mmdet/custom/ocp.py)
- Only the eventual Occlusion Copy & Paste implementation is provided in this repository for brevity. If you require implementation codes for other add-ons (realism enhancers as described in our paper), please contact us.  
- Training config files for OCP can be found in: 
  - Mask-RCNN, R50 backbone, FPN:
    - Vanilla Baseline (w/o copy paste): [`configs/mask_rcnn/coco_human-vanilla_baseline-75epochs.py`](configs/mask_rcnn/coco_human-vanilla_baseline-75epochs.py)
    - Basic Copy & Paste: [`configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py`](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py)
    - Occluded Copy & Paste: [`configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py`](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py)
  - Mask2Former, Swin-S backbone: 
    - Vanilla Baseline (w/o copy paste): [`configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune.py`](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune.py)
    - Occluded Copy & Paste: [`configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py`](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py)
- Batch sizes in training configs are for our training hardware: 1 node with 4 x NVIDIA V100 GPUs

### (additional) Pose2Seg implementation 

- OCP is additionally implementated for training with [Pose2Seg](https://github.com/liruilong940607/Pose2Seg). 
- Please refer to [./Pose2Seg_OCP/README.md](./Pose2Seg_OCP/README.md). 
- Pose2Seg is implemented in plain PyTorch, so our OC&P code implementation there should be more suitable for any new adaptations elsewhere that don't use MMDetection. 
- Main efforts in implementing OC&P there is additionally handling copy & pasting and augmentation of keypoints (which wasn't considered in Mask RCNN or Mask2Former).


## Datasets

### Expected structure

(modify accordingly)

```
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
    ├── ochuman_coco_format_val_range_0.00_1.00_full_labelled.json
    └── ochuman_coco_format_test_range_0.00_1.00_full_labelled.json
```

### Dataset Download

- COCO 2017 can be downloaded [here](https://cocodataset.org/#download). 
- OCHuman can be downloaded [here](https://github.com/liruilong940607/OCHumanApi).

### OCHuman Fully Labelled

OCHuman Fully Labelled (FL) is introduced in our paper for fairer evaluation, as [original OCHuman](https://github.com/liruilong940607/OCHumanApi) val & test sets contain images that are not exhaustively labelled (with ground-truth masks). We provide here a subset of the OCHuman sets that contain exhaustively labelled ones. This subset is derived from OCHuman's original json label files (`ochuman.json`), where they exhaustively label the bounding boxes for most of the human instances (of course there are ones that are missed out, but mostly negligible), but only selectively label pose keypoints & masks. This subset contains images that have bounding boxes with labelled GT masks. All credits of images and annotations go to OCHuman creators. JSON files of this subset: 
  - [ochuman_coco_format_val_range_0.00_1.00_full_labelled.json]()
  - [ochuman_coco_format_test_range_0.00_1.00_full_labelled.json]()

## Train

- We train with multi-gpu [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Look into said bash script to change your paths and namings accordingly

```
./tools/dist_train.sh
```

- NOTE: for Mask2Former trainings, we finetune from pre-trained COCO weights, so download the [corresponding weight files](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former) from MMDetection and update the `load_from=` paths in the corresponding config files.

## Evaluation

- Look into said bash script to change your paths and namings accordingly

```
cd occlusion-copy-paste/
./tools/dist_test.sh
```

### Weights download

- Download weights from links in [results table below](#results), and store in [`./weights/`](./weights/)

## Results

See more details and info in paper.


|         Model                  |  OCHuman AP Val | OCHuman AP Test | OCHuman(FL) AP Val | OCHuman(FL) AP Test | Config |   Weights   |
| :----------------------------: | :-------------: | :-------------: | :----------------: | :-----------------: | :----: | :---------: |
|   [Pose2Seg](https://arxiv.org/abs/1803.10683) |         -       |       -         |        22.8        |        22.9         |   -    |  [from their repo](https://github.com/liruilong940607/Pose2Seg) |
|  + Occlusion C&P (ours)        |         -       |       -         |        25.3        |        25.1         |   -    |  [gdrive dl link]() |
|   [Mask R-CNN](https://arxiv.org/abs/1703.06870) (pretrained)      |       14.9      |      14.9       |        24.5        |        24.9         |   [from mmdet](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py)    |  [from mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn) |
|   Mask R-CNN (vanilla trained) |       16.5      |      16.6       |        27.0        |        27.4         |   [cfg](configs/mask_rcnn/coco_human-vanilla_baseline-75epochs.py)    |  -  |
|  + Basic C&P (ours)            |       18.6      |      17.8       |        29.3        |        28.5         |   [cfg](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py)    |  [gdrive dl link](https://drive.google.com/uc?confirm=t&id=1wE0wwPDfkSBJjdStaXySCVFazFhVUnRv) |
|  + Occlusion C&P (ours)        |       19.5      |      18.6       |        30.6        |        29.9         |   [cfg](configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py)    |  [gdrive dl link](https://drive.google.com/uc?confirm=t&id=1VdZfaK8Ck79RtYn6FDqAS3o_2kpINauc) |
|   [Mask2Former](https://arxiv.org/abs/2112.01527) (pretrained)     |       25.9      |      25.4       |        43.2        |        44.7         |   [from mmdet](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py)    |  [from mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former) |
|   Mask2Former (vanilla trained)|       26.7      |      26.3       |        45.2        |        46.4         |   [cfg](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune.py)    |  -  |
|  + [Simple Copy-Paste](https://arxiv.org/abs/2012.07177)           |       28.0      |      27.7       |        48.9        |        50.2         |   [cfg](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-SCP-5e.py)   |  - |
|  + Occlusion C&P (ours)        |       28.9      |      28.3       |        49.3        |        50.6         |   [cfg](configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py)    |  [gdrive dl link](https://drive.google.com/uc?confirm=t&id=1K48JBMgQlWM2z7g3rFslfbX_KIw8KH-o) |

