
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import numpy as np
import cv2
import pose_input

slim = tf.contrib.slim


# In[2]:

LABEL_SIZE = 46

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

def inference(images,weight_decay,reuse=None):
    """Build a human pose model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Logits.
    """
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm, # apply batchnorm after each conv layer
                        normalizer_params=batch_norm_params):
        # stage 0
        net0 = slim.conv2d(images, 16, [3, 3], scope='conv0_stage0')
        net0 = slim.conv2d(net0, 16, [3, 3], scope='conv1_stage0')
        net0 = slim.conv2d(net0, 16, [3, 3], rate=2, scope='conv2_stage0')
        net0 = slim.max_pool2d(net0, [2, 2], scope='pool0_stage0')

        net0 = slim.conv2d(net0, 32, [3, 3], scope='conv3_stage0')
        net0 = slim.conv2d(net0, 32, [3, 3], scope='conv4_stage0')
        net0 = slim.conv2d(net0, 32, [3, 3], rate=2, scope='conv5_stage0')
        net0 = slim.max_pool2d(net0, [2, 2], scope='pool1_stage0')

        net0 = slim.conv2d(net0, 64, [3, 3], scope='conv6_stage0')
        net0 = slim.conv2d(net0, 64, [3, 3], scope='conv7_stage0')
        net0 = slim.conv2d(net0, 64, [3, 3], rate=2, scope='conv8_stage0')
        net0 = slim.max_pool2d(net0, [2, 2], scope='pool2_stage0')    

        net0 = slim.conv2d(net0, 128, [3, 3], scope='conv9_stage0')
        net0 = slim.conv2d(net0, 128, [3, 3], scope='conv10_stage0')
        net0 = slim.conv2d(net0, 128, [3, 3], rate=2, scope='conv11_stage0')
        net0 = slim.conv2d(net0, 128, [3, 3], rate=2, scope='conv12_stage0')
        net0 = slim.conv2d(net0, 128, [1, 1], scope='fc0_stage0')
        net0 = slim.conv2d(net0, pose_input.NUM_HEATMAPS, [1, 1], scope='fc1_stage0')

        net1 = slim.conv2d(images, 16, [3, 3], scope='conv0_stage1')
        net1 = slim.conv2d(net1, 16, [3, 3], scope='conv1_stage1')
        net1 = slim.conv2d(net1, 16, [3, 3], rate=2, scope='conv2_stage1')
        net1 = slim.max_pool2d(net1, [2, 2], scope='pool0_stage1')

        net1 = slim.conv2d(net1, 32, [3, 3], scope='conv3_stage1')
        net1 = slim.conv2d(net1, 32, [3, 3], scope='conv4_stage1')
        net1 = slim.conv2d(net1, 32, [3, 3], rate=2, scope='conv5_stage1')
        net1 = slim.max_pool2d(net1, [2, 2], scope='pool1_stage1')
        
        net1 = slim.conv2d(net1, 64, [3, 3], scope='conv6_stage1')
        net1 = slim.conv2d(net1, 64, [3, 3], scope='conv7_stage1')
        net1 = slim.conv2d(net1, 64, [3, 3], rate=2, scope='conv8_stage1')
        net1 = slim.max_pool2d(net1, [2, 2], scope='pool2_stage1')
        
        net1 = slim.conv2d(net1, 128, [3, 3], scope='conv9_stage1')
        net1_conv10 = slim.conv2d(net1, 32, [3, 3], scope='conv10_stage1')
        
        net1 = tf.concat(3,[net1_conv10,net0])
        net1 = slim.conv2d(net1, 128, [3, 3], scope='conv11_stage1')
        net1 = slim.conv2d(net1, 128, [3, 3], scope='conv12_stage1')
        net1 = slim.conv2d(net1, 128, [3, 3], rate=2, scope='conv13_stage1')
        net1 = slim.conv2d(net1, 128, [3, 3], rate=4, scope='conv14_stage1')
        net1 = slim.conv2d(net1, 128, [3, 3], rate=8, scope='conv15_stage1')
        net1 = slim.conv2d(net1, 128, [1, 1], scope='fc0_stage1')
        net1 = slim.conv2d(net1, pose_input.NUM_HEATMAPS, [1, 1], scope='fc1_stage1')
        
    return net0,net1

def loss(heatmaps_stage0, heatmaps_stage1, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs().
    Returns:
        Loss tensor of type float.
    """
    # labels0 contain heatmaps that include other people's joints
    # They are used for training local detectors
    # labels1 contain heatmaps that only include self joints
    # They are used for contextual stages of training
    resized_labels = tf.image.resize_images(labels,[LABEL_SIZE,LABEL_SIZE])
    labels1,labels0 = tf.split(3, 2, resized_labels)
    error0 = slim.losses.mean_squared_error(heatmaps_stage0, labels0)
    slim.losses.add_loss(error0)
    error1 = slim.losses.mean_squared_error(heatmaps_stage1, labels1)
    slim.losses.add_loss(error1)
    # The total loss is defined as the Euclidean loss plus all of the weight
    # decay terms (L2 loss).
    return slim.losses.get_total_loss()


# In[3]:

# DIRECTORY = '/Users/xuehan.xiong/Google Drive/datasets/human_pose'
# TFRECORD_FILE = os.path.join(DIRECTORY, 'pose_small.tfrecords')

# file_path = os.path.join(DIRECTORY,TFRECORD_FILE)
# images,labels = pose_input.distorted_inputs([file_path],32,1000)
# heatmaps0,heatmaps1 = inference(images,0.0)
# heatmap_loss = loss(heatmaps0, heatmaps1, labels)
# init = tf.initialize_all_variables()

# sess = tf.InteractiveSession()
# sess.run(init)
# # Start the queue runners.
# tf.train.start_queue_runners(sess=sess)
# loss_val = sess.run(heatmap_loss)


# In[4]:

# print loss_val


# In[ ]:



