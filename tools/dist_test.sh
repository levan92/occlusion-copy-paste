#!/usr/bin/env bash

GPUS=4 #number of gpus per node
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# config="configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-basic_copy_paste.py"
# checkpoint="weights/coco_human-mask_rcnn_r50_fpn-basic_copy_paste-ep25.pth"

config="configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py"
checkpoint="weights/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste-ep24.pth"

# config="configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e.py"
# checkpoint="weights/mask2former_swin-s-p4-w7-224_lsj_4x1_50e_coco-person-finetune-OCP_aug-5e-iter48000.pth"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $config \
    $checkpoint \
    --launcher pytorch \
    --fuse-conv-bn \
    --eval bbox segm 