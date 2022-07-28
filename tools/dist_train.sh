#!/usr/bin/env bash

set -e 

GPUS=4 #number of gpus per node
PORT=${PORT:-29500}

WORK_DIR_PARENT="/path/to/your/workdir/"

config_dir="configs/mask2former"
run_name="coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste" # say, for running `configs/mask_rcnn/coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste.py`

run_name="${run_name%.py}"
echo "Running training for $run_name.."

WORK_DIR="${WORK_DIR_PARENT}/${run_name}"

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $config_dir/${run_name}.py --work-dir  ${WORK_DIR} --launcher pytorch ${@:3}

echo "############ Training is done! ############"

echo "Testing.."
$(dirname "$0")/test_after_train.sh ${run_name} ${WORK_DIR} ${GPUS} ${PORT}
echo "########## Tests done! ###########"


