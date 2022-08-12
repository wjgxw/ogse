from networks.ops import *
flags = tf.flags
flags.DEFINE_integer('MAX_INTER', 150000, 'The number of trainning steps')
flags.DEFINE_integer('MAX_TO_KEEP', 1, 'The max number of model to save')
flags.DEFINE_integer('BATCH_SIZE', 8, 'The size of batch images [16]')
flags.DEFINE_float('BETA', 1e-1, 'Learning rate of for Optimizer [8e-2]')
flags.DEFINE_integer(
    'STEP', None,
    'Which checkpoint should be load, None for final step of checkpoint [None]'
)
flags.DEFINE_float('LR', 1e-4, 'Learning rate of for Optimizer [1e-4]')
flags.DEFINE_integer('NUM_GPUS', 1, 'The number of GPU to use [1]')
flags.DEFINE_boolean('IS_TRAIN', True, 'True for train, else test. [True]')
flags.DEFINE_boolean(
    'LOAD_MODEL', True,
    'True for load checkpoint and continue training. [True]'
)
flags.DEFINE_integer(
    'FILTER_DIM', 64,
    'The number of feature maps in all layers. [64]'
)
flags.DEFINE_string(
    'MODEL_DIR', 'DenseNet_vin',
    'If LOAD_MODEL, provide the MODEL_DIR. [./model/baseline/]'
)
flags.DEFINE_string(
    'DATA_DIR', '/data4/angus_wj/ogse/ogse_deep3/data4train1/',
    'the data set direction'
)
FLAGS = flags.FLAGS


def model(inpt, reuse = False, name='zhangjun'):
    with tf.variable_scope('inpute_layer', reuse=reuse):
        conv_in = conv_relu(inpt, [3, 3, FLAGS.INPUT_C, FLAGS.FILTER_DIM], 1)

    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            res1 = residual_block(conv_in, FLAGS.FILTER_DIM)
        with tf.variable_scope('res2'):
            res2 = residual_block(res1, FLAGS.FILTER_DIM)
        with tf.variable_scope('conv1'):
            conv_add1 = tf.concat((conv_in, res1, res2), 3)
            conv1 = conv_relu(conv_add1, [3, 3, FLAGS.FILTER_DIM*3, FLAGS.FILTER_DIM], 1)
        with tf.variable_scope('res3'):
            res3 = residual_block(conv1, FLAGS.FILTER_DIM)
        with tf.variable_scope('res4'):
            res4 = residual_block(res3, FLAGS.FILTER_DIM)
        with tf.variable_scope('conv2'):
            conv_add2 = tf.concat((conv1, res3, res4), 3)
            conv2 = conv_relu(conv_add2, [3, 3, FLAGS.FILTER_DIM*3, FLAGS.FILTER_DIM], 1)
        with tf.variable_scope('res5'):
            res5 = residual_block(conv2, FLAGS.FILTER_DIM)
        with tf.variable_scope('res6'):
            res6 = residual_block(res5, FLAGS.FILTER_DIM)
        with tf.variable_scope('conv3'):
            conv_add3 = tf.concat((conv2, res5, res6), 3)
            conv3 = conv_relu(conv_add3, [3, 3, FLAGS.FILTER_DIM*3, FLAGS.FILTER_DIM], 1)
        with tf.variable_scope('res7'):
            res7 = residual_block(conv3, FLAGS.FILTER_DIM)
        with tf.variable_scope('res8'):
            res8 = residual_block(res7, FLAGS.FILTER_DIM)
        with tf.variable_scope('conv4'):
            conv_add4 = tf.concat((conv3, res7, res8), 3)
            conv4 = conv_relu(conv_add4, [3, 3, FLAGS.FILTER_DIM*3, FLAGS.FILTER_DIM], 1)

    with tf.variable_scope('output_layer'):
        conv_out = conv_relu(conv4, [3, 3, FLAGS.FILTER_DIM, FLAGS.OUTPUT_C], 1)
    return conv_out

def losses(output, labels, name = 'losses'):
    with tf.name_scope(name):
        # change1 = labels <= 0.05
        # change1 = tf.cast(change1, tf.float32)
        # Y1 = change1 * 0.05
        #
        # change2 = labels > 0.05
        # change2 = tf.cast(change2, tf.float32)
        # Y2 = tf.multiply(labels, change2)
        #
        # Y_change = Y1 + Y2
        # loss = (tf.reduce_mean(tf.square(logits - labels)/Y_change)+
        #      FLAGS.BETA * tf.reduce_mean(TotalVariation(logits))
        # )
        loss = tf.norm(output[:, :, :, :] - labels[:, :, :, :],1)
        #loss = tf.reduce_mean(tf.square(output - labels))
        return loss, loss

