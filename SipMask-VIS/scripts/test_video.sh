#!/bin/bash

python tools/test_video.py configs/sipmask/sipmask_r50_caffe_fpn_gn_1x.py checkpoints/vis_sipmask_ms_1x_final.pth --out results/results.pkl --eval segm --show --save_path results/images 
