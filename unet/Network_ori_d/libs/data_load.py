#-*- coding:utf-8 -*-
from os.path import exists,join
import os
from skimage import io
import tensorflow as tf
import numpy as np

from utils import *

def make_datasets_from_memory(datasets,labelsets,epoch,batch_size,data_num=None, preprocess=None):
    if data_num == None:
        data_num = batch_size*10
    dataset = tf.data.Dataset.from_tensor_slices((datasets,labelsets))
    if not preprocess==None:
        dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(data_num).repeat(epoch).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x,y = iterator.get_next()
    return x,y

def make_datasets_from_mat(datapath,labelpath,epoch,batch_size,data_num=None, preprocess=None):
    if data_num == None:
        data_num = batch_size*10
    images = listdir_abspath(datapath)
    labels = listdir_abspath(labelpath)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.map(
        lambda images, labels: tuple(tf.py_func(
            mat_to_tensor, [images, labels], [tf.double, tf.double])), num_parallel_calls=4)

    # pre-process
    if not preprocess==None:
        dataset = dataset.map(preprocess,num_parallel_calls=4)

    #produce batch
    dataset = dataset.shuffle(data_num)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()

    return x,y

def make_datasets_from_image(datapath, labelpath, epoch, batch_size, mulitch=False, data_num=None, preprocess=None):
    '''
    A mulit-channels dataset shouled be like this:
        --datapath
            --1
                --data1.png
                --data2.png
                --...
            --2
                --data1.png
                --data2.png
                --...
        --labelpath
                --data1.png
                --data2.png
                --...
    '''
    if data_num == None:
        data_num = batch_size*10

    #single-channel data
    if not mulitch:
        data_list = listdir_abspath(datapath)
        label_list = listdir_abspath(labelpath)
        images = tf.constant(data_list)
        labels = tf.constant(label_list)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(fn_to_tensor,num_parallel_calls=4)

    # mulit-channels data
    else:
        data_list = listdir_abspath_mulitch(datapath)  # [[1.png],[2.png],[3.png]...]
        label_list = listdir_abspath(labelpath)  # [1.png, 2.png, 3.png...]
        images = tf.constant(data_list)
        labels = tf.constant(label_list)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(fn_to_tensor_mulitch, num_parallel_calls=4)

    #pre-process
    if not preprocess == None:
        dataset = dataset.map(preprocess, num_parallel_calls=4)

    #produce batch
    dataset = dataset.shuffle(data_num).repeat(epoch).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x,y = iterator.get_next()
    return x,y

def make_datasets_from_tfrecord(datapath, epoch, batch_size, data_num=None, preprocess=None):
    if data_num == None:
        data_num = batch_size*10

    dataset = tf.data.TFRecordDataset(datapath)  # load tfrecord file
    dataset = dataset.map(tfrecord_reader, num_parallel_calls=4)  # parse data into tensor

    # pre-process
    if not preprocess == None:
        dataset = dataset.map(preprocess, num_parallel_calls=4)

    # produce batch
    dataset = dataset.shuffle(data_num)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    return x,y

def make_testsets_from_mat(datapath, crop_size=None, label=False):
    data_mat = matio.loadmat(datapath)['data']

    if not label:
        data_mat = np.expand_dims(data_mat, axis=0)

    if not crop_size==None:
        data_mat = tensor_to_patch(data_mat, crop_size)
    #out = tf.cast(out, format)
    return data_mat

if __name__ == "__main__":
    import time
    datapath = 'data/train_data.tfrecords'
    labelpath = 'data/train/label'
    test_datapath = 'data/test/test_data_1.mat'

    # datapath = 'F:\Python_code\OLED_8_Rec\oled_four_echo\data\demo_data\single\data'
    # labelpath = 'F:\Python_code\OLED_8_Rec\oled_four_echo\data\demo_data\single\label'
    def pro_fn(x,y):
        x = tf.random_crop(x, [64, 64, 8],seed=1)
        y = tf.random_crop(y, [64, 64, 1],seed=1)
        #x = tf.image.random_flip_left_right(x)
        # x = tf.cast(x,tf.float32)
        # y = tf.cast(y,tf.float32)
        return x, y

    # x = make_datasets_from_mat(datapath, labelpath, epoch=2, batch_size=16, preprocess=pro_fn)
    test = make_testsets_from_mat(test_datapath, crop_size=64)
    x = make_datasets_from_tfrecord(datapath, epoch=2, batch_size=16, preprocess=pro_fn)

    with tf.Session() as sess:
        try:
            while True:
                time_start = time.time()
                xx = sess.run(x)
                print(xx[0].shape,xx[1].shape,test.shape)
                print('iteration')
                time_end = time.time()
                print('totally cost', time_end - time_start)
        except tf.errors.OutOfRangeError:
            print("end!")