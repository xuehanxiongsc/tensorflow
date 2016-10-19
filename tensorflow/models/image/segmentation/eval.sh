#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/portrait_segmentation_test.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation/portrait_segmentation_test.tfrecords'
CUDA_VISIBLE_DEVICES=1 python segmentation_eval.py \
--tfrecord_path="$TFRECORD_PATH_SERVER" \
--batch_size=64 \
--eval_interval_secs=120
