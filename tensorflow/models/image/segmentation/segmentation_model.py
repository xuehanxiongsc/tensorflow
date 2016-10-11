import tensorflow as tf
import numpy as np

SEED = None
WEIGHT_DECAY = 0.00001
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                          "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
      return tf.float16
  else:
      return tf.float32

def _variable_with_weight_decay(shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
  Returns:
      Variable Tensor
  """
  var = tf.Variable(tf.truncated_normal(
          shape,
          stddev=stddev, 
          dtype=data_type()))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def BatchNorm(inputT, is_training=True, scope=None):
  return tf.contrib.layers.batch_norm(inputT, activation_fn=tf.nn.relu, is_training=True,  
          center=False, updates_collections=None, scope=scope)
  # Note: is_training is tf.placeholder(tf.bool) type
  #return tf.cond(is_training,  
  #    lambda: tf.contrib.layers.batch_norm(inputT, activation_fn=tf.nn.relu, is_training=True,  
  #        center=False, updates_collections=None, scope=scope),  
  #    lambda: tf.contrib.layers.batch_norm(inputT, activation_fn=tf.nn.relu, is_training=False,  
  #        updates_collections=None, center=False, scope=scope, reuse = True))  

def inference(images, is_train):
  """Build a segmentation model.
  Args:
      images: Images returned from distorted_inputs() or inputs().
  Returns:
      Logits.
  """
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,3,8],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([8], dtype=data_type()))
    conv1 = tf.nn.bias_add(conv, biases, name='conv1')

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool1')
  
  # batch norm1
  batch_norm1 = BatchNorm(pool1, is_train)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,8,16],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([16], dtype=data_type()))
    conv2 = tf.nn.bias_add(conv, biases, name='conv2')

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool2')
  
  # batch norm2
  batch_norm2 = BatchNorm(pool2, is_train)
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,16,32],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([32], dtype=data_type()))
    conv3 = tf.nn.bias_add(conv, biases, name='conv3')

  # pool3
  pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool3')
  
  # batch norm3
  batch_norm3 = BatchNorm(pool3, is_train)
  
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,32,64],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    conv4 = tf.nn.bias_add(conv, biases, name='conv4')
  
  # batch norm4
  batch_norm4 = BatchNorm(conv4, is_train)
  
  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,64,64],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([64], dtype=data_type()))
    conv5 = tf.nn.bias_add(conv, biases, name='conv5')
  
  # batch norm5
  batch_norm5 = BatchNorm(conv5, is_train)
  
  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay(
        shape=[3,3,64,128],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([128], dtype=data_type()))
    conv6 = tf.nn.bias_add(conv, biases, name='conv6')
  
  # batch norm6
  batch_norm6 = BatchNorm(conv6, is_train)
  
  # fc, implemented as conv
  with tf.variable_scope('fc') as scope:
    kernel = _variable_with_weight_decay(
        shape=[1,1,128,4],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    conv = tf.nn.conv2d(batch_norm6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([4], dtype=data_type()))
    fc = tf.nn.bias_add(conv, biases, name='fc')
      
  # deconv
  with tf.variable_scope('deconv') as scope:
    kernel = _variable_with_weight_decay(
        shape=[16,16,4,4],
        stddev=0.01,
        wd=WEIGHT_DECAY)
    images_dims = images.get_shape().as_list()
    output_shape = [images_dims[0],images_dims[1],images_dims[2],4]
    strides = [1,8,8,1]
    deconv = tf.nn.conv2d_transpose(fc, kernel, output_shape, strides, padding='SAME',name='deconv')

  return deconv

def loss_and_accuracy(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs().
  Returns:
      Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int32)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  
  _,indices = tf.nn.top_k(logits, k=1, sorted=False)
  squeezed_indices = tf.squeeze(indices)
  acc = tf.contrib.metrics.accuracy(squeezed_indices,labels)
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss'),acc


