#!/usr/bin/env bash
set -e 

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

bn=${1}
echo "Testing ${bn}.."

WORK_DIR=${2}
echo "Work dir: ${WORK_DIR}"

GPUS=${3:-4}
echo "Num gpus: ${GPUS}"

PORT=${4:-29500}
echo "Port: ${PORT}"

best_model=`ls ${WORK_DIR}/best_1_segm_mAP_*.pth -t | head -1`
echo "best model: ${best_model}"
cfg=`ls ${WORK_DIR}/*.py -rt | head -1`
echo "config file: ${cfg}"

test_set="test-best"
echo "testing on ${test_set}"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    ${cfg} \
    ${best_model} \
    --launcher pytorch \
    --fuse-conv-bn \
    --work-dir ${WORK_DIR}/${test_set}/ \
    --out ${WORK_DIR}/${test_set}/results.pkl \
    --eval bbox segm \
    --show \
    --show-dir ${WORK_DIR}/${test_set}/viz/ \
    --show-score-thr 0.3 \
    --eval-options jsonfile_prefix=${WORK_DIR}/${test_set}/res_jsons/res
    