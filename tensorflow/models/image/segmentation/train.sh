#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/portrait_segmentation_train.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation/portrait_segmentation_train.tfrecords'
CUDA_VISIBLE_DEVICES=2 python segmentation_train.py \
--tfrecord_path="$TFRECORD_PATH_SERVER" \
--max_steps=10000 \
--decay_steps=3000 \
--weight_decay=0.001
