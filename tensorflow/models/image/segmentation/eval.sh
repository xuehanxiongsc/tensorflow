#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/portrait_segmentation.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation/portrait_segmentation_test.tfrecords'
python segmentation_eval.py \
--tfrecord_path="$TFRECORD_PATH_LOCAL" \
--batch_size=64 \
--device_type='CPU' \
--device_id='5,6' \
--eval_interval_secs=120
