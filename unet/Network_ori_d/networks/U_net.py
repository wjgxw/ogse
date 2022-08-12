from networks.ops import *
import tensorflow as tf
import numpy as np
#paramaters
flags = tf.flags
flags.DEFINE_integer('MAX_INTER', 150000, 'The number of trainning steps')
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
    'LOAD_MODEL', True,
    'True for load checkpoint and continue training. [True]'
)
flags.DEFINE_string(
    'MODEL_DIR', 'U_Net_d',
    'If LOAD_MODEL, provide the MODEL_DIR. [./model/baseline/]'
)
flags.DEFINE_string(
    'DATA_DIR', '/data4/angus_wj/ogse/ogse_deep4/data4train/',
    'the data set directihtopon'
)
FLAGS = flags.FLAGS

#deep 5
def model(images, reuse = False, name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_1 = conv_relu(images, [3, 3, FLAGS.INPUT_C, FLAGS.FILTER_DIM], 1)
        L1_2 = conv_relu(L1_1, [3, 3, 64, FLAGS.FILTER_DIM], 1)
        L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')  ##

        L2_2 = conv_relu(L2_1, [3, 3, FLAGS.FILTER_DIM, FLAGS.FILTER_DIM*2], 1)
        L2_3 = conv_relu(L2_2, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*2], 1)
        L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##

        L3_2 = conv_relu(L3_1, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*4], 1)
        L3_3 = conv_relu(L3_2, [3, 3, FLAGS.FILTER_DIM*4, FLAGS.FILTER_DIM*4], 1)
        L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##

        L4_2 = conv_relu(L4_1, [3, 3, FLAGS.FILTER_DIM*4, FLAGS.FILTER_DIM*8], 1)
        L4_3 = conv_relu(L4_2, [3, 3, FLAGS.FILTER_DIM*8, FLAGS.FILTER_DIM*8], 1)
        L5_1 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L5_2 = conv_relu(L5_1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 16], 1)
        L5_3 = conv_relu(L5_2, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 16], 1)

        L4_U1 = deconv2(L5_3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8, 2, 2)
        L4_U1 = tf.concat((L4_3, L4_U1), 3)
        L4_U2 = conv_relu(L4_U1, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8], 1)
        L4_U3 = conv_relu(L4_U2, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 8], 1)

        L3_U1 = deconv2(L4_U3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4, 2, 2)
        L3_U1 = tf.concat((L3_3, L3_U1), 3)
        L3_U2 = conv_relu(L3_U1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4], 1)
        L3_U3 = conv_relu(L3_U2, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 4], 1)

        L2_U1 = deconv2(L3_U3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2, 2, 2)
        L2_U1 = tf.concat((L2_3, L2_U1), 3)
        L2_U2 = conv_relu(L2_U1, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2], 1)
        L2_U3 = conv_relu(L2_U2, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 2], 1)

        L1_U1 = deconv2(L2_U3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1, 2, 2)
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = conv_relu(L1_U1, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1], 1)
        L1_U3 = conv_relu(L1_U2, [3, 3, FLAGS.FILTER_DIM * 1, FLAGS.FILTER_DIM * 1], 1)

        out = conv_relu(L1_U3, [3, 3, FLAGS.FILTER_DIM, FLAGS.OUTPUT_C], 1)

    # variables = tf.contrib.framework.get_variables(name)

    return out

#deep 3
##def model(images, reuse = False, name='UNet'):
 #   with tf.variable_scope(name, reuse=reuse):
##        L1_1 = conv_relu(images, [3, 3, FLAGS.INPUT_C, FLAGS.FILTER_DIM], 1)
 #       L1_2 = conv_relu(L1_1, [3, 3, 64, FLAGS.FILTER_DIM], 1)
  #      L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')  ##
#
  #      L2_2 = conv_relu(L2_1, [3, 3, FLAGS.FILTER_DIM, FLAGS.FILTER_DIM*2], 1)
  #      L2_3 = conv_relu(L2_2, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*2], 1)
  #      L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##
#
  #      L3_2 = conv_relu(L3_1, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*4], 1)
  #      L3_3 = conv_relu(L3_2, [3, 3, FLAGS.FILTER_DIM*4, FLAGS.FILTER_DIM*4], 1)
  #      L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

  #      L4_2 = conv_relu(L4_1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 16], 1)
  #      L4_3 = conv_relu(L4_2, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 16], 1)

  #      L3_U1 = deconv2(L4_U3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4, 2, 2)
 #       L3_U1 = tf.concat((L3_3, L3_U1), 3)
 #       L3_U2 = conv_relu(L3_U1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4], 1)
 #       L3_U3 = conv_relu(L3_U2, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 4], 1)

   #     L2_U1 = deconv2(L3_U3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2, 2, 2)
  #      L2_U1 = tf.concat((L2_3, L2_U1), 3)
  #      L2_U2 = conv_relu(L2_U1, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2], 1)
  #      L2_U3 = conv_relu(L2_U2, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 2], 1)

  #      L1_U1 = deconv2(L2_U3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1, 2, 2)
   #     L1_U1 = tf.concat((L1_2, L1_U1), 3)
  #      L1_U2 = conv_relu(L1_U1, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1], 1)
  #      L1_U3 = conv_relu(L1_U2, [3, 3, FLAGS.FILTER_DIM * 1, FLAGS.FILTER_DIM * 1], 1)
#
 #       out = conv_relu(L1_U3, [3, 3, FLAGS.FILTER_DIM, FLAGS.OUTPUT_C], 1)

    # variables = tf.contrib.framework.get_variables(name)

  #  return out

def losses(output, labels, name = 'losses'):
    with tf.name_scope(name):
        loss_T2 = tf.norm(output[:, :, :, :] - labels[:, :, :, :],1)
        #loss_T2 = tf.reduce_mean(tf.square(output - labels))

        loss = loss_T2
        return loss, loss_T2

# def losses_CCB(output, labels, name = 'losses_CCB'):
#     with tf.name_scope(name):
#         mask = labels[:, :, :, 0]
#         ychange = labels[:, :, :, 1]
#         loss1 = (tf.reduce_mean(tf.square(output[:, :, :, 0] - labels[:, :, :, 2]) * ychange ) + 2*FLAGS.BETA * tf.reduce_mean(TotalVariation(output[:, :, :, 0]) * mask))  #T2 loss
#         return loss1, loss1
# Medical Image Synthesis with Deep convolutional adversarial network,TBME,2018
# def losses(output, labels, alpha=1):
#     """
#     Calculates the sum of GDL losses between the predicted and ground truth images.
#     @param gen_CT: The predicted CTs.
#     @param gt_CT: The ground truth images
#     @param alpha: The power to which each gradient term is raised.
#     @param batch_size_tf batch size
#     @return: The GDL loss.
#     """
#     # calculate the loss for each scale
#
#     # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
#     pos = tf.constant(np.identity(1), dtype=tf.float32)
#     neg = -1 * pos
#     filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
#     filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
#     strides = [1, 1, 1, 1]  # stride of (1, 1)
#     padding = 'SAME'
#
#     gen_dx = tf.abs(tf.nn.conv2d(tf.expand_dims(output[:,:,:,1],-1), filter_x, strides, padding=padding))
#     gen_dy = tf.abs(tf.nn.conv2d(tf.expand_dims(output[:,:,:,1],-1), filter_y, strides, padding=padding))
#     gt_dx = tf.abs(tf.nn.conv2d(tf.expand_dims(labels[:,:,:,1],-1), filter_x, strides, padding=padding))
#     gt_dy = tf.abs(tf.nn.conv2d(tf.expand_dims(labels[:,:,:,1],-1), filter_y, strides, padding=padding))
#
#     grad_diff_x = tf.abs(gt_dx - gen_dx)
#     grad_diff_y = tf.abs(gt_dy - gen_dy)
#
#     gdl=1e-5*tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha))
#
#
#     # loss_tv = 1e-2*tf.reduce_mean(TotalVariation(logits))
#     loss_T2 = tf.reduce_mean(tf.square(output[:, :, :, 0] - labels[:, :, :, 0]))
#     loss_T2star = tf.reduce_mean(tf.square(output[:, :, :, 1] - labels[:, :, :, 1]))
#     loss = loss_T2 + loss_T2star
#     return loss, loss_T2

