
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


# In[2]:

SEED = None
WEIGHT_DECAY = 0.00001
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


# In[3]:

def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32

def inference(images,weight_decay):
    """Build a segmentation model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """    
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': ['moving_vars'],
            'moving_variance': ['moving_vars'],
        }
    }

    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm, # apply batchnorm after each conv layer
                        normalizer_params=batch_norm_params):
        
        net = slim.conv2d(images, 8, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        
        net = slim.conv2d(net, 16, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        
        net = slim.conv2d(net, 32, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')    

        net = slim.conv2d(net, 64, [3, 3], scope='conv4')
        net = slim.conv2d(net, 64, [3, 3], scope='conv5')
        net = slim.conv2d(net, 128, [3, 3], scope='conv6')
        
        net = slim.conv2d(net, 4, [1, 1], scope='fc')
    
    net = slim.conv2d_transpose(net, 4, [16, 16], 8, 
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                biases_initializer=None,
                                activation_fn=None,
                                padding='VALID',
                                scope='deconv')
    return net

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
    logits_shape = logits.get_shape().as_list()
    label_shape = labels.get_shape().as_list()
    height = label_shape[1]
    width  = label_shape[2]
    num_classes = logits_shape[3]
    predictions = tf.slice(logits,[0,4,4,0],[label_shape[0],height,width,num_classes],name='crop')
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    slim.losses.add_loss(cross_entropy_mean)
    
    _,indices = tf.nn.top_k(predictions, k=1, sorted=False)
    squeezed_indices = tf.squeeze(indices)
    acc = tf.contrib.metrics.accuracy(squeezed_indices,labels)
    tf.scalar_summary('accuracy', acc)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return slim.losses.get_total_loss(),acc


# In[ ]:



