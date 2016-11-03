#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/MPI_test.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/human_pose/MPI_test.tfrecords'
CUDA_VISIBLE_DEVICES=3 python pose_eval.py \
--tfrecord_path="$TFRECORD_PATH_SERVER" \
--batch_size=16 \
--eval_interval_secs=120
