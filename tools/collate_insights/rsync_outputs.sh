#!/bin/bash

vm=${1:-"vm1"}
remote_dir="/datadisk/levan/improved-instance-segm/models/"
local_dir="/Users/levan/Workspace/improved-instance-segm/outputs/"

# from local MBP
rsync -varP --exclude "*.tar.gz"  --exclude "*.log" --exclude "*.log.json"  --exclude "*.segm.json" --exclude "*.bbox.json"  --exclude "*.pth" --exclude "*.jpg"  --exclude "*.pkl"  ${vm}:${remote_dir} ${local_dir}