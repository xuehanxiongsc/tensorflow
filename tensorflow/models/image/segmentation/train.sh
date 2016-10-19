#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/portrait_segmentation_train.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation/portrait_segmentation_train.tfrecords'
CUDA_VISIBLE_DEVICES=2 python segmentation_train.py \
--tfrecord_path="$TFRECORD_PATH_SERVER" \
--max_steps=12000 \
--batch_size=64 \
--decay_steps=4000 \
--weight_decay=0.00001 \
--save_interval_secs=120 \
--initial_learning_rate=0.1
