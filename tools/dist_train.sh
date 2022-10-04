#!/usr/bin/env bash

set -e 

GPUS=4 #number of gpus per node
PORT=${PORT:-29500}

WORK_DIR_PARENT="./work_dirs/"

config_dir="configs/mask_rcnn/"
# run_name="coco_human-mask_rcnn_r50_fpn-basic_copy_paste" 
run_name="coco_human-mask_rcnn_r50_fpn-occlusion_copy_paste" #config file with/without suffix '.py'


run_name="${run_name%.py}" # this will strip trailing .py
echo "Running training for $run_name.."

WORK_DIR="${WORK_DIR_PARENT}/${run_name}"

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $config_dir/${run_name}.py --work-dir  ${WORK_DIR} --launcher pytorch ${@:3}

echo "############ Training is done! ############"

echo "Testing.."
$(dirname "$0")/test_after_train.sh ${run_name} ${WORK_DIR} ${GPUS} ${PORT}
echo "########## Tests done! ###########"


