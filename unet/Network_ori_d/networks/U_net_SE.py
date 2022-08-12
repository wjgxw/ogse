from networks.ops import *
import tensorflow as tf
import numpy as np

# paramaters
flags = tf.flags
flags.DEFINE_integer('MAX_INTER', 200000, 'The number of trainning steps')
flags.DEFINE_integer('MAX_TO_KEEP', 1, 'The max number of model to save')
flags.DEFINE_integer('BATCH_SIZE', 8, 'The size of batch images [16]')
flags.DEFINE_float('BETA', 1e-6, 'TV Optimizer [8e-2]')
flags.DEFINE_integer(
    'STEP', None,
    'Which checkpoint should be load, None for final step of checkpoint [None]'
)
flags.DEFINE_float('LR', 1e-4, 'Learning rate of for Optimizer [1e-4]')
flags.DEFINE_integer('NUM_GPUS', 1, 'The number of GPU to use [1]')
flags.DEFINE_boolean('IS_TRAIN', True, 'True for train, else test. [True]')
flags.DEFINE_integer(
    'FILTER_DIM', 64,
    'The number of feature maps in all layers. [64]'
)
flags.DEFINE_boolean(
    'LOAD_MODEL', False,
    'True for load checkpoint and continue training. [True]'
)
flags.DEFINE_string(
    'MODEL_DIR', 'UNet_T2_9.22_5',
    'If LOAD_MODEL, provide the MODEL_DIR. [./model/baseline/]'
)
flags.DEFINE_string(
    'DATA_DIR', '/data3/wj/T2star/datagen50/data4trainccb/',
    'the data set directihtopon'
)
FLAGS = flags.FLAGS
reduction_ration = 4

# deep 5
def model(images, reuse=False, name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_1 = conv_relu(images, [3, 3, FLAGS.INPUT_C, FLAGS.FILTER_DIM], 1)
        L1_1 = Squeeze_excitation_layer(L1_1, FLAGS.FILTER_DIM, reduction_ration, 'SE1')
        L1_2 = conv_relu(L1_1, [3, 3, 64, FLAGS.FILTER_DIM], 1)
        L1_2 = Squeeze_excitation_layer(L1_2, FLAGS.FILTER_DIM, reduction_ration, 'SE2')
        L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L2_2 = conv_relu(L2_1, [3, 3, FLAGS.FILTER_DIM, FLAGS.FILTER_DIM * 2], 1)
        L2_2 = Squeeze_excitation_layer(L2_2, FLAGS.FILTER_DIM*2, reduction_ration, 'SE3')
        L2_3 = conv_relu(L2_2, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 2], 1)
        L2_3 = Squeeze_excitation_layer(L2_3, FLAGS.FILTER_DIM * 2, reduction_ration, 'SE4')
        L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L3_2 = conv_relu(L3_1, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 4], 1)
        L3_2 = Squeeze_excitation_layer(L3_2, FLAGS.FILTER_DIM * 4, reduction_ration, 'SE5')
        L3_3 = conv_relu(L3_2, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 4], 1)
        L3_3 = Squeeze_excitation_layer(L3_3, FLAGS.FILTER_DIM * 4, reduction_ration, 'SE6')
        L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L4_2 = conv_relu(L4_1, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 8], 1)
        L4_2 = Squeeze_excitation_layer(L4_2, FLAGS.FILTER_DIM * 8, reduction_ration, 'SE7')
        L4_3 = conv_relu(L4_2, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 8], 1)
        L4_3 = Squeeze_excitation_layer(L4_3, FLAGS.FILTER_DIM * 8, reduction_ration, 'SE8')
        L5_1 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L5_2 = conv_relu(L5_1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 16], 1)
        L5_2 = Squeeze_excitation_layer(L5_2, FLAGS.FILTER_DIM * 16, reduction_ration, 'SE9')
        L5_3 = conv_relu(L5_2, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 16], 1)
        L5_3 = Squeeze_excitation_layer(L5_3, FLAGS.FILTER_DIM * 16, reduction_ration, 'SE10')

        L4_U1 = deconv2(L5_3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8, 2, 2)
        L4_U1 = tf.concat((L4_3, L4_U1), 3)
        L4_U2 = conv_relu(L4_U1, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8], 1)
        L4_U2 = Squeeze_excitation_layer(L4_U2, FLAGS.FILTER_DIM * 8, reduction_ration, 'SE11')
        L4_U3 = conv_relu(L4_U2, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 8], 1)
        L4_U3 = Squeeze_excitation_layer(L4_U3, FLAGS.FILTER_DIM * 8, reduction_ration, 'SE12')


        L3_U1 = deconv2(L4_U3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4, 2, 2)
        L3_U1 = tf.concat((L3_3, L3_U1), 3)
        L3_U2 = conv_relu(L3_U1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4], 1)
        L3_U2 = Squeeze_excitation_layer(L3_U2, FLAGS.FILTER_DIM * 4, reduction_ration, 'SE13')
        L3_U3 = conv_relu(L3_U2, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 4], 1)
        L3_U3 = Squeeze_excitation_layer(L3_U3, FLAGS.FILTER_DIM * 4, reduction_ration, 'SE14')


        L2_U1 = deconv2(L3_U3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2, 2, 2)
        L2_U1 = tf.concat((L2_3, L2_U1), 3)
        L2_U2 = conv_relu(L2_U1, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2], 1)
        L2_U2 = Squeeze_excitation_layer(L2_U2, FLAGS.FILTER_DIM * 2, reduction_ration, 'SE15')
        L2_U3 = conv_relu(L2_U2, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 2], 1)
        L2_U3 = Squeeze_excitation_layer(L2_U3, FLAGS.FILTER_DIM * 2, reduction_ration, 'SE16')


        L1_U1 = deconv2(L2_U3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1, 2, 2)
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = conv_relu(L1_U1, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1], 1)
        L1_U2 = Squeeze_excitation_layer(L1_U2, FLAGS.FILTER_DIM * 1, reduction_ration, 'SE17')
        L1_U3 = conv_relu(L1_U2, [3, 3, FLAGS.FILTER_DIM * 1, FLAGS.FILTER_DIM * 1], 1)
        L1_U3 = Squeeze_excitation_layer(L1_U3, FLAGS.FILTER_DIM * 1, reduction_ration, 'SE18')


        out = conv_relu(L1_U3, [3, 3, FLAGS.FILTER_DIM, FLAGS.OUTPUT_C], 1)

    # variables = tf.contrib.framework.get_variables(name)

    return out


def losses(output, labels, name='losses'):
    with tf.name_scope(name):
        #loss = tf.reduce_mean(tf.square((output - labels)))
        loss = tf.norm(output - labels,1)
        return loss, loss


def losses_CCB(output, labels, name='losses_CCB'):
    with tf.name_scope(name):
        mask = labels[:, :, :, 0]
        ychange = labels[:, :, :, 1]
        loss1 = (tf.reduce_mean(
            tf.square(output[:, :, :, 0] - labels[:, :, :, 2]) * ychange) + 2 * FLAGS.BETA * tf.reduce_mean(
            TotalVariation(output[:, :, :, 0]) * mask))  # T2 loss
        return loss1, loss1
