
# coding: utf-8

# In[1]:

import tensorflow as tf
import math
import numpy as np
import os
from datetime import datetime
import time
import numpy as np
import pose_input
import pose_model
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from six.moves import xrange
slim = tf.contrib.slim


# In[2]:

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25307
INITIAL_LEARNING_RATE = 1.0E-1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9
MOMENTUM = 0.9
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfrecord_path', '',
                           """Path to TF record """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/pose_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('weight_decay', 0.00001,
                            """Weight decay factor.""")
tf.app.flags.DEFINE_integer('decay_epochs', 10,
                            """Number of epoches per learning rate decay.""")
tf.app.flags.DEFINE_integer('max_epochs', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            """Time interval to save checkpoints.""")
tf.app.flags.DEFINE_integer('log_every_n_steps', 10,
                            """Steps per logging.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
steps_per_epoch = int(math.ceil(float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)/float(FLAGS.batch_size)))
decay_steps = FLAGS.decay_epochs*steps_per_epoch
max_steps = FLAGS.max_epochs*steps_per_epoch


# In[ ]:

def train():
    with tf.Graph().as_default():
        images, labels = pose_input.distorted_inputs(
            [FLAGS.tfrecord_path],
            FLAGS.batch_size,
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        heatmap0 = pose_model.inference(images,FLAGS.weight_decay)

        # Calculate loss.
        loss = pose_model.loss(heatmap0, labels)
        
        # create a global step variable
        global_step = variables.get_or_create_global_step()
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
        tf.scalar_summary('learning_rate',lr)
        # updates the model parameters.
        opt = tf.train.MomentumOptimizer(lr,MOMENTUM,use_nesterov=True)
        train_op = slim.learning.create_train_op(loss, optimizer=opt)    
        slim.learning.train(train_op,FLAGS.train_dir,
                           log_every_n_steps=FLAGS.log_every_n_steps,
                           save_interval_secs=FLAGS.save_interval_secs,
                           number_of_steps=max_steps)
        
def main(unused_args):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    train()
        
if __name__ == "__main__":
    tf.app.run()

