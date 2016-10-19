
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import math
from datetime import datetime
import time
import numpy as np
import segmentation_input
import segmentation_model
slim = tf.contrib.slim


# In[2]:

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 4

tf.app.flags.DEFINE_string('tfrecord_path', '',
                           """Path to TF records.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/segmentation_train/',
                           """Directory where to load checkpoints.""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/segmentation_eval',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 120,
                            """The minimum number of seconds between evaluations.""")
NUM_TEST_SAMPLES = 800
NUM_EVAL=int(math.ceil(float(NUM_TEST_SAMPLES)/float(FLAGS.batch_size)))


# In[5]:

def evaluate():
    with tf.Graph().as_default():
        images, labels = segmentation_input.distorted_inputs(
            [FLAGS.tfrecord_path],
            FLAGS.batch_size,
            NUM_TEST_SAMPLES)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = segmentation_model.inference(images,0.0)
        logits_shape = logits.get_shape().as_list()
        label_shape = labels.get_shape().as_list()
        height = label_shape[1]
        width  = label_shape[2]
        num_classes = logits_shape[3]
        predictions = tf.slice(logits,[0,4,4,0],[label_shape[0],height,width,num_classes],name='crop')
        
        _,predicted_labels = tf.nn.top_k(predictions, k=1, sorted=False)
        predicted_labels = tf.squeeze(predicted_labels)
        
        mean_iou,update_op = slim.metrics.streaming_mean_iou(predicted_labels,labels,NUM_CLASSES)
        
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



