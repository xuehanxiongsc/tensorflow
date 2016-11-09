
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import math
from datetime import datetime
import time
import numpy as np
import pose_input
import pose_model
import cv2
slim = tf.contrib.slim


# In[2]:

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecord_path', '/Users/xuehan.xiong/Google Drive/datasets/human_pose/MPI_test.tfrecords',
                           """Path to TF records.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/pose_train/',
                           """Directory where to load checkpoints.""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/pose_eval',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 120,
                            """The minimum number of seconds between evaluations.""")
NUM_TEST_SAMPLES = 2500
NUM_EVAL=int(math.ceil(float(NUM_TEST_SAMPLES)/float(FLAGS.batch_size)))
num_heatmaps = pose_input.NUM_HEATMAPS


# In[5]:

with tf.Graph().as_default():
    images, labels = pose_input.distorted_inputs(
            [FLAGS.tfrecord_path],
            FLAGS.batch_size,
            NUM_TEST_SAMPLES)
    heatmaps = pose_model.inference(images,0.0)
    heatmaps = tf.sigmoid(heatmaps)
    heatmaps0,heatmaps1 = tf.split(3,2,heatmaps)
    labels0,labels1 = tf.split(3,2,labels)
    resized_labels = tf.image.resize_images(labels1,
                                            [pose_model.LABEL_SIZE,pose_model.LABEL_SIZE])
    
    mean_iou,update_op = slim.metrics.streaming_mean_squared_error(heatmaps1,resized_labels)    
    iou = slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=NUM_EVAL,
        eval_op=update_op,
        final_op=mean_iou,
        eval_interval_secs=FLAGS.eval_interval_secs)


# In[6]:

def main(unused_args):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    evaluate()
    
if __name__ == "__main__":
    tf.app.run()


# In[ ]:



