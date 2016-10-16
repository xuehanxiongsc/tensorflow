
# coding: utf-8

# In[4]:

import cv2
import tensorflow as tf
import numpy as np
import os
import random
import scipy.misc as misc

#sess = tf.InteractiveSession()


# In[2]:

DIRECTORY = '/Users/xuehan.xiong/Google Drive/datasets/selfies_segmentation'
IMAGE_DIRECTORY = os.path.join(DIRECTORY,'selfies')
LABEL_DIRECTORY = os.path.join(DIRECTORY,'gt_selfies')
TOLERANCE = 10
RESIZE_DIM = 128

def is_image(file):
    if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg")     or file.endswith(".png") or file.endswith(".JPEG"):
        return True
    return False

def get_data(image_dir,label_dir):
    images = np.empty(0,dtype=object)
    labels = np.empty(0,dtype=object)
    for filename in os.listdir(image_dir):
        image_file = os.path.join(image_dir,filename)
        if not is_image(filename):
            continue
        label_file = os.path.join(label_dir,filename)
        if not os.path.isfile(label_file):
            continue
        images = np.append(images,image_file)
        labels = np.append(labels,label_file)
    return images, labels

def split_train_test(images,labels,test_split):
    total_samples = images.size
    train_size = int((1.0-test_split)*total_samples)
    x = [[i] for i in range(total_samples)]
    random.shuffle(x)
    images_train = np.empty(0,dtype=object)
    labels_train = np.empty(0,dtype=object)
    images_test = np.empty(0,dtype=object)
    labels_test = np.empty(0,dtype=object)
    train_x = x[:train_size]
    test_x  = x[train_size:]
    for i in train_x:
        images_train = np.append(images_train,images[i])
        labels_train = np.append(labels_train,labels[i])
    for i in test_x:
        images_test = np.append(images_test,images[i])
        labels_test = np.append(labels_test,labels[i])
    return images_train,labels_train,images_test,labels_test

def red_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,0],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,1],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,2],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def green_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,1],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,0],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,2],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def blue_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,2],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,0],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,1],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _gen_label_image(image):
    hair_mask = red_mask(image)
    face_mask = green_mask(image)
    shoulder_mask = blue_mask(image)
    label_image = np.zeros_like(image[:,:,0])
    label_image[hair_mask==255] = 1
    label_image[shoulder_mask==255] = 2
    label_image[face_mask==255] = 3
    kernel_width = np.maximum(int(label_image.shape[0]/100.0),5)
    kernel = np.ones((kernel_width,kernel_width),np.uint8)
    label_image = cv2.morphologyEx(label_image, cv2.MORPH_CLOSE, kernel)
    return label_image

def _resize_image_label(image,label):
    height,width = image.shape[:2]
    scale = float(RESIZE_DIM) / float(np.minimum(width,height))
    resized_image = cv2.resize(image,(0,0),fx=scale,fy=scale)
    resized_label = cv2.resize(label,(resized_image.shape[1],resized_image.shape[0]),interpolation=cv2.INTER_NEAREST)
    assert resized_image.shape[:2] == resized_label.shape[:2]
    assert np.minimum(resized_image.shape[0],resized_image.shape[1]) == RESIZE_DIM
    return resized_image,resized_label

def convert_to(image_files,label_files,name):
    num_examples = image_files.size
    print('Total of %d images' % num_examples)
    if label_files.size != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                     (num_examples, label_files.size))

    output = os.path.join(DIRECTORY, name + '.tfrecords')
    print('Writing', output)
    writer = tf.python_io.TFRecordWriter(output)
    for index in range(num_examples):
        # read image
        # print image_files[index]
        image_i = misc.imread(image_files[index])
        assert image_i.ndim == 3
        rows = image_i.shape[0]
        cols = image_i.shape[1]
        if image_i.shape[2] > 3:
            image_i = image_i[:,:,:3]
        depth = image_i.shape[2]
        assert depth == 3
        # read label
        label_i = misc.imread(label_files[index])
        label_i = _gen_label_image(label_i)
        # resize image and label
        resized_image, resized_label = _resize_image_label(image_i,label_i)
        assert np.amax(resized_label) <= 3
        image_raw = resized_image.tostring()
        label_raw = resized_label.tostring()
        # create an example in TF's format
        example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(resized_image.shape[0]),
        'width': _int64_feature(resized_image.shape[1]),
        'depth': _int64_feature(depth),
        'label': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


# In[5]:

image_files,label_files = get_data(IMAGE_DIRECTORY,LABEL_DIRECTORY)
images_train,labels_train,images_test,labels_test = split_train_test(image_files,label_files,0.1)
convert_to(images_train,labels_train,'portrait_segmentation_train')
convert_to(images_test,labels_test,'portrait_segmentation_test')




# In[ ]:



