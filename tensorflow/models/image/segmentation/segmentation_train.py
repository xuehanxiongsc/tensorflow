
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
from datetime import datetime
import time
import numpy as np
import segmentation_input
import segmentation_model
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from six.moves import xrange
slim = tf.contrib.slim


# In[2]:

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 7195
NUM_EPOCHS_PER_DECAY = 10
INITIAL_LEARNING_RATE = 1.0E-1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9
MOMENTUM = 0.9
STEPS_PER_DISPLAY = 10
STEPS_PER_SUMMARY = 100
STEPS_PER_CHECKPT = 1000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfrecord_path', '',
                           """Path to TF record """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/segmentation_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('weight_decay', 0.00001,
                            """Weight decay factor.""")
tf.app.flags.DEFINE_integer('decay_steps', 2000,
                            """Number of steps per learning rate decay.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            """Time interval to save checkpoints.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


# In[5]:

def train():
    with tf.Graph().as_default():
        images, labels = segmentation_input.distorted_inputs(
            [FLAGS.tfrecord_path],
            FLAGS.batch_size,
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = segmentation_model.inference(images,FLAGS.weight_decay)

        # Calculate loss.
        loss,acc = segmentation_model.loss_and_accuracy(logits, labels)
        
        # create a global step variable
        global_step = variables.get_or_create_global_step()
        tf.Print(global_step,[global_step])
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
        
        tf.scalar_summary('learning_rate',lr)
        # updates the model parameters.
        # opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
        opt = tf.train.MomentumOptimizer(lr,MOMENTUM,use_nesterov=True)
        train_op = slim.learning.create_train_op(loss, optimizer=opt)    
        slim.learning.train(train_op,FLAGS.train_dir,
                           log_every_n_steps=10,
                           save_interval_secs=FLAGS.save_interval_secs,
                           number_of_steps=FLAGS.max_steps)


# In[6]:

def main(unused_args):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    train()
        
if __name__ == "__main__":
    tf.app.run()


# In[ ]:



