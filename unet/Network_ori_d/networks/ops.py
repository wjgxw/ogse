# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
#from tflearn.layers.conv import global_avg_pool

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv_prelu(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    pre_relu = tf.nn.bias_add(conv, biases)
    out = prelu(pre_relu)

    return out


def conv_bn_relu(inpt, filter_shape, stride, is_training):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    batch_norm = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
    out = tf.nn.relu(batch_norm)

    return out


def conv_bn(inpt, filter_shape, stride, is_training):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    batch_norm = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)

    return batch_norm


def conv_relu(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    pre_relu = tf.nn.bias_add(conv, biases)
    out = tf.nn.relu(pre_relu)

    return out


def conv(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    out = tf.nn.bias_add(conv, biases)

    return out

def deconv2(x, input_filter, output_filter, kernel, strides):
    with tf.variable_scope('conv_transpose'):
        shape = [kernel, kernel, output_filter, input_filter]
        weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = 'weight')
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]*strides
        width = tf.shape(x)[2]*strides
        output_shape = tf.stack([batch_size, height, width, output_filter])
    return tf.nn.conv2d_transpose(x, weight, output_shape, strides = [1, strides, strides,1], name = 'conv_transpose')


def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg


def residual_block(inpt, output_depth):
    input_depth = inpt.get_shape().as_list()[3]

    with tf.variable_scope('conv1_in_block'):
        conv1 = conv_relu(inpt, [3, 3, input_depth, output_depth], 1)

    with tf.variable_scope('conv2_in_block'):
        conv2 = conv(conv1, [3, 3, output_depth, output_depth], 1)

    res = conv2 + inpt
    out = tf.nn.relu(res)

    return out


def TotalVariation(images, name = None):
    '''
    Calculate and return the total variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the
    images.

    This can be used as a loss-function during optimization so as to suppress
    noise in images. If you have a batch of images, then you should calculate
    the scalar loss-value as the sum:
    `loss = tf.reduce_sum(tf.image.total_variation(images))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or
                3-D Tensor of shape `[height, width, channels]`.

        name: A name for the operation (optional).

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.

        If `images` was 4-D, return a 1-D float Tensor of shape `[batch]`
        with the total variation for each image in the batch.
        If `images` was 3-D, return a scalar float with the total variation for
        that image.
    '''

    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 3:

            pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
            pixel_dif1 = array_ops.pad(
                pixel_dif1, paddings = [[0,1], [0,0], [0,0]]
            )
            pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
            pixel_dif2 = array_ops.pad(
                pixel_dif2, paddings = [[0,0], [0,1], [0,0]]
            )

        elif ndims == 4:

            pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
            pixel_dif1 = array_ops.pad(
                pixel_dif1, paddings = [[0,0], [0,1], [0,0], [0,0]]
            )
            pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
            pixel_dif2 = array_ops.pad(
                pixel_dif2, paddings = [[0,0], [0,0], [0,1], [0,0]]
            )

        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences
        tot_var = math_ops.abs(pixel_dif1) + math_ops.abs(pixel_dif2)

    return tot_var
def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

    return scale

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)
