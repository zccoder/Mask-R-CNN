#!/usr/bin/env bash



python ../src/autopilot/train_autopilot.py \
--CUDA_VISIBLE_DEVICES=8,9 \
--dataset=/home1/testt/cvpr/data/image_output \
--annotation_file=/home1/testt/cvpr/data/annotations/instances_all_shuffle.json \
--model=/home1/testt/cvpr/mrcnn/mask_rcnn_coco.h5

#--model=/home1/testt/cvpr/mrcnn/mask_rcnn_coco.h5
#--CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
