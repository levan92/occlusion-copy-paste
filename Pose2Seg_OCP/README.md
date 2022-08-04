# OCP implementation in Pose2Seg

This sub-repo holds Occlusion Copy & Paste implementation for Pose2Seg training code. Our original Occlusion Copy & Paste implemented in `mmdetection` is re-implemented in plain PyTorch here. 

We provide here the modified and additional files required to train Pose2Seg with OC&P, on top of their original repository: [Pose2Seg](https://github.com/liruilong940607/Pose2Seg).

## Testing without Pose Keypoint GTs

Pose2Seg uses [Associative Embedding Pose Estimation](https://github.com/princeton-vl/pose-ae-train) to predict keypoints as proposals into Pose2Seg model. We use the same repo to generate predicted keypoints and convert the predicted outputs into COCO json format to feed into Pose2Seg for testing. Script for conversion is at [`Pose2Seg_OCP/aepose/aedets2cocojson.py`](Pose2Seg_OCP/aepose/aedets2cocojson.py).