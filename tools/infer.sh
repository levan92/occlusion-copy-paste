#!/bin/bash 

VIDEO=/path/to/video/file.mp4

CONFIG=/path/to/config.py

WEIGHTS=/path/to/weights.pth

OUT=/path/to/output.mp4

OUTFRAME=/path/to/output/frames/

python3 $(dirname "$0")/infer.py \
    $VIDEO \
    $CONFIG \
    $WEIGHTS \
    --score-thr 0.3 \
    --outframes $OUTFRAME \
    # --out $OUT \
