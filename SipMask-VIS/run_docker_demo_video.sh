#!/bin/bash

#docker run --gpus all --shm-size=8g -it -v $(pwd)/checkpoints:/mmdetection/checkpoints -v $(pwd)/configs:/mmdetection/configs -v $(pwd)/results:/mmdetection/results -v $(pwd)/tools:/mmdetection/tools mmdetection:latest
docker run --gpus all --shm-size=8g -it \
-v $(pwd)/checkpoints:/mmdetection/SipMask-VIS/checkpoints \
-v $(pwd)/configs:/mmdetection/SipMask-VIS/configs \
-v $(pwd)/data:/mmdetection/SipMask-VIS/data \
-v $(pwd)/demo:/mmdetection/SipMask-VIS/demo \
-v $(pwd)/scripts:/mmdetection/SipMask-VIS/scripts \
-v $(pwd)/results:/mmdetection/SipMask-VIS/results \
-v $(pwd)/pycocotools:/mmdetection/SipMask-VIS/pycocotools \
-v $(pwd)/tools:/mmdetection/SipMask-VIS/tools \
-v $(pwd)/mmdet:/mmdetection/SipMask-VIS/mmdet \
mmdetection:sipmask-vis \
python demo/video.py
#ls -la
