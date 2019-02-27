#!/usr/bin/env bash



python ../src/autopilot/predict.py \
--CUDA_VISIBLE_DEVICES=5 \
--image_dir=/home2/data/zc/driver/test \
--model=/home1/testt/cvpr/mrcnn/mask_rcnn_coco.h5
