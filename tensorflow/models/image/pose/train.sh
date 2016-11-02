#!/bin/bash
TFRECORD_PATH_SERVER='/raid/xuehan/MPI_train.tfrecords'
TFRECORD_PATH_LOCAL='/Users/xuehan.xiong/Google Drive/datasets/human_pose/MPI_train.tfrecords'
CUDA_VISIBLE_DEVICES=2 python pose_train.py \
--tfrecord_path="$TFRECORD_PATH_SERVER" \
--max_epochs=20 \
--batch_size=16 \
--decay_epochs=10 \
--weight_decay=0.00001 \
--save_interval_secs=600 \
--initial_learning_rate=0.01
