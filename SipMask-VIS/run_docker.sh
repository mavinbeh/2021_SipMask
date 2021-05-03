#!/bin/bash

#docker run --gpus all --shm-size=8g -it -v $(pwd)/checkpoints:/mmdetection/checkpoints -v $(pwd)/configs:/mmdetection/configs -v $(pwd)/results:/mmdetection/results -v $(pwd)/tools:/mmdetection/tools mmdetection:latest
docker run --gpus all --shm-size=8g -it \
-v $(pwd)/checkpoints:/SipMask-VIS/checkpoints \
-v $(pwd)/configs:/SipMask-VIS/configs \
-v $(pwd)/data:/SipMask-VIS/data \
-v $(pwd)/demo:/SipMask-VIS/demo \
-v $(pwd)/scripts:/SipMask-VIS/scripts \
-v $(pwd)/results:/SipMask-VIS/results \
-v $(pwd)/pycocotools:/SipMask-VIS/pycocotools \
-v $(pwd)/tools:/SipMask-VIS/tools \
-v $(pwd)/mmdet:/SipMask-VIS/mmdet \
mavinbeh/mmdetection:sipmask-vis \
scripts/test_video.sh
#ls -la data
