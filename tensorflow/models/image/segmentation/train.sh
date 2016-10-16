#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/portrait_segmentation.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation/portrait_segmentation_train.tfrecords'
python segmentation_train.py \
--tfrecord_path="$TFRECORD_PATH_LOCAL" \
--max_steps=3000
