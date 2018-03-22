"""
https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
"""


from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import batch_norm


he_init = layers.variance_scaling_initializer()



def conv(inputs, num_filter, ksize, stride, padding='SAME', is_training=True, scope=None):
    bottom_shape = inputs.get_shape().as_list()[3]

    with tf.variable_scope(scope):
        W = tf.get_variable("W", [ksize, ksize, bottom_shape, num_filter],
                            initializer=he_init)

        x = tf.nn.conv2d(inputs, W, strides=[1, stride, stride, 1],
                         padding=padding)
        x = layers.batch_norm(x, is_training=is_training,
                              decay=0.9, updates_collections=None)
        x = tf.nn.relu(x)

    return x



def conv_preAct(inputs, num_filter, ksize, stride, padding, is_training, scope=None):
    bottom_shape = inputs.get_shape().as_list()[3]

    with tf.variable_scope(scope):
        bn = layers.batch_norm(inputs, is_training=is_training,
                              decay=0.9, updates_collections=None)

        act = tf.nn.relu(bn)

        W = tf.get_variable("W", [ksize, ksize, bottom_shape, num_filter],
                            initializer=he_init)

        out = tf.nn.conv2d(act, W, strides=[1, stride, stride, 1],
                         padding=padding)



    return out


def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_preAct(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_preAct(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_preAct(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    return res

def resnet(input,is_train):
    conv_1 = conv(input, 64, 7, 2, "SAME", is_train, 'conv_1')
    pool_1 = maxpool(conv_1, 3, 2, "SAME", 'pool_1')

    res_out = pool_1
    for i in range(3):
        res_1_0 = conv_preAct(res_out, 64, 3, 1, "SAME", is_train, 'res_1_0' + str(i))
        res_1_1 = conv_preAct(res_1_0, 64, 3, 1, "SAME", is_train, 'res_1_1' + str(i))
        res_out = res_out + res_1_1

    res_2_0 = conv_preAct(res_out, 128, 3, 2, "SAME", is_train, 'res_2_0' )
    res_2_1 = conv_preAct(res_2_0, 128, 3, 1, "SAME", is_train, 'res_2_1' )

    pool_2 = conv_preAct(res_out, 128, 1, 2, "SAME", is_train, 'pool_2')
    res_out = pool_2 + res_2_1

    for i in range(3):
        res_2_0 = conv_preAct(res_out, 128, 3, 1, "SAME", is_train, 'res_2_0' + str(i))
        res_2_1 = conv_preAct(res_2_0, 128, 3, 1, "SAME", is_train, 'res_2_1' + str(i))
        res_out = res_out + res_2_1

    avg_pool = tf.nn.avg_pool(res_out, ksize=[1, 4, 4, 1],
                          strides=[1, 1, 1, 1],
                          padding="VALID")
    rst = fc(avg_pool, 10, 'logits')
    return rst

def test(input, is_training):
    conv1 = conv(input, 16, ksize=7, stride=2,
                 padding="SAME", is_training=is_training, scope="conv1")
    conv1 = maxpool(conv1, ksize=2, stride=2,
                    padding="SAME", scope="pool1")

    conv2 = conv(conv1, 32, ksize=3, stride=1,
                 padding="SAME", is_training=is_training, scope="conv2")
    conv2 = maxpool(conv2, ksize=2, stride=2,
                    padding="SAME", scope="pool2")

    conv3 = conv(conv2, 64, ksize=3, stride=1,
                 padding="SAME", is_training=is_training, scope="conv3")
    conv3 = maxpool(conv3, ksize=2, stride=2,
                    padding="SAME", scope="pool3")

    fc1 = fc_relu(conv3, 128, is_training=is_training, scope="fc1")
    logit = fc(fc1, 10, scope="logit")
    return logit

def maxpool(inputs, ksize, stride, padding, scope=None):
    with tf.variable_scope(scope):
        pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool

def fc(inputs, num_dims, scope=None):
    bottom_shape = inputs.get_shape().as_list()
    if len(bottom_shape) > 2:
        inputs = tf.reshape(inputs,
                            [-1, reduce(lambda x, y: x * y, bottom_shape[1:])])
        bottom_shape = inputs.get_shape().as_list()

    with tf.variable_scope(scope):
        W = tf.get_variable("W", [bottom_shape[1], num_dims],
                            initializer=he_init)
        b = tf.get_variable("b", [num_dims],
                            initializer=tf.constant_initializer(0))

        out = tf.matmul(inputs, W) + b
    return out


def fc_relu(inputs, num_dims, is_training, scope=None):
    with tf.variable_scope(scope):
        out = fc(inputs, num_dims, scope="fc")
        bn = layers.batch_norm(out, is_training=is_training,
                               decay=0.9, updates_collections=None)
        relu = tf.nn.relu(bn)
    return relu

