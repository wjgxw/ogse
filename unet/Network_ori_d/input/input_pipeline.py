import numpy as np
import tensorflow as tf
from glob import glob
import os.path as osp

flags = tf.flags
flags.DEFINE_integer('NUM_EPOCHS', 10000, 'The number of train epochs [4e3]')
flags.DEFINE_integer('INPUT_H', 128, 'The hight of input image [128]')
flags.DEFINE_integer('INPUT_W', 128, 'The width of input image [128]')
flags.DEFINE_integer('INPUT_C', 16, 'The channel of input image [2]')
flags.DEFINE_integer('OUTPUT_C', 1, 'The channel of output image [2]')     #
flags.DEFINE_integer('LABEL_C', 4, 'The channel of output image [2]')     #
flags.DEFINE_integer('CROP_X', 16, 'The hight of crop box [32]')
flags.DEFINE_integer('CROP_Y', 16, 'The width of crop box [32]')
FLAGS = flags.FLAGS


def inpute(data_dir, type='train'):
    print("loading %s data ......" % type)
    if type == 'train':
        data_paths = glob(osp.join(data_dir, type, '*.Charles'))
    elif type == 'test':
        data_paths = glob(osp.join(data_dir, type, '*.Charles'))
    else:
        data_paths = sorted(glob(osp.join(data_dir, type, '*.Charles')))

    sample_num = len(data_paths)
    if sample_num==0:
        raise RuntimeError('There is no data in this direction')
    input_sets = np.zeros((sample_num, FLAGS.INPUT_H, FLAGS.INPUT_W, FLAGS.INPUT_C), dtype=np.float32)
    label_sets = np.zeros((sample_num, FLAGS.INPUT_H, FLAGS.INPUT_W, FLAGS.OUTPUT_C), dtype=np.float32)
    for fileidx in range(sample_num):
        data_in = np.fromfile(data_paths[fileidx], dtype=np.float32)
        data_pairs = data_in.reshape(FLAGS.INPUT_H, FLAGS.INPUT_W, FLAGS.INPUT_C+FLAGS.LABEL_C)
        input_sets[fileidx, :, :, :] = data_pairs[:, :, :FLAGS.INPUT_C]
        label_sets[fileidx, :, :, :] = data_pairs[:, :, 16:17] # d 16:17  vin 17:18  Dex 18:19
    print("finished\n")

    return input_sets, label_sets



def process (read_data, read_labels):

    batch = FLAGS.BATCH_SIZE                                                                                      #每次输入图片个数
    size = FLAGS.CROP_X

    data_cases, data_height, data_width, data_channels = read_data.shape
    labels_cases, labels_height, labels_width, labels_channels = read_labels.shape

    rand_index = np.random.random_integers(0, data_cases - 1, size=batch)
    rand_index.sort()
    data = read_data[rand_index, :, :, :]
    labels = read_labels[rand_index, :, :, :]

    crops_x = np.random.random_integers(0, high=data_height - size, size=batch)
    crops_y = np.random.random_integers(0, high=data_width - size, size=batch)
    random_cropped_data = np.zeros((batch, size, size, data_channels), dtype=np.float32)
    random_cropped_labels = np.zeros((batch, size, size, labels_channels), dtype=np.float32)

    for i in range(batch):
        random_cropped_data[i, :, :, :] = data[i, crops_x[i]: (crops_x[i] + size), crops_y[i]: (crops_y[i] + size), :]
        random_cropped_labels[i, :, :, :] = labels[i, (crops_x[i]):(crops_x[i] + size), (crops_y[i]):(crops_y[i] + size), :]

    return random_cropped_data, random_cropped_labels

