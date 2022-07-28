#!/usr/bin/env bash

GPUS=4 #number of gpus per node
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

name="coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste"
ckpt_name="best_0_segm_mAP_iter_112000.pth"

WORK_DIR_PARENT="/path/to/your/workdir/"
WORK_DIR="${WORK_DIR_PARENT}/${run_name}"

test_dirname="test-best"

test_dir=${WORK_DIR}/${test_dirname}
CONFIG="${WORK_DIR}/${name}.py"
CHECKPOINT="${WORK_DIR}/${ckpt_name}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --fuse-conv-bn \
    --eval bbox segm \
    --show \
    --show-dir ${test_dir}/viz/ \
    --show-score-thr 0.3 \
    --work-dir ${test_dir}/ \
    --out ${test_dir}/results.pkl \
    --eval-options jsonfile_prefix=${test_dir}/res_jsons/res
