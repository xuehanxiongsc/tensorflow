
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
get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots


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
joint_names = ['R_ankle','R_knee','R_hip','L_hip','L_knee','L_ankle','R_wrist',
               'R_elbow','R_shoulder','L_shoulder','L_elbow','L_wrist','neck',
               'head','bgd']


# In[5]:

def blend_heatmap(image,heatmap):
    norm = plt.Normalize(vmax=np.amax(heatmap))
    output = plt.cm.jet(norm(heatmap))*255.0;
    output = output[:,:,:3]
    output = cv2.resize(output,(image.shape[1], image.shape[0]))
    output = output*0.5 + image*0.5
    return output.astype(np.uint8)
    
def plot_heatmaps(image, heatmaps, offset):
    f, axarr = plt.subplots(4, 4)
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    for i in xrange(4):
        for j in xrange(4):
            if i == 0 and j == 0:
                axarr[i,j].imshow(image)
                axarr[i,j].set_title("Image")
            else:
                blended_image = blend_heatmap(image,heatmaps[:,:,offset+4*i+j-1])
                axarr[i,j].imshow(blended_image)
                axarr[i,j].set_title(joint_names[4*i+j-1])
            axarr[i,j].axis('off')
    plt.show()
    
with tf.Graph().as_default():
    images, labels = pose_input.distorted_inputs(
            [FLAGS.tfrecord_path],
            FLAGS.batch_size,
            NUM_TEST_SAMPLES)
    heatmaps = pose_model.inference(images,0.0)
    heatmaps = tf.sigmoid(heatmaps)
#     labels0,labels1 = tf.split(3, 2, labels)
#     resized_labels = tf.image.resize_images(labels1,
#                                             [pose_model.LABEL_SIZE,pose_model.LABEL_SIZE])
    saver = tf.train.Saver()
    sess = tf.Session()
    
    model_checkpoint_path = os.path.join(FLAGS.checkpoint_dir,'model.ckpt-15820')
    saver.restore(sess, model_checkpoint_path)
    tf.train.start_queue_runners(sess=sess)
    images_val,heatmaps_val = sess.run([images,heatmaps])


# In[22]:

index = 0
imagei = (images_val[index,:,:,:3] + 0.5)*255.0
imagei = imagei.astype(np.uint8)
blue = np.copy(imagei[:,:,0])
imagei[:,:,0] = imagei[:,:,2]
imagei[:,:,2] = blue;
plot_heatmaps(imagei, heatmaps_val[index,:,:,:], 15)


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



