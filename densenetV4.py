# -*- coding: utf-8 -*-
import tensorflow as tf
from cbam1d import se_block, cbam_block

def Bottleneck(x,growthRate,kernel_size):
    network = tf.layers.batch_normalization(inputs=x)
    network = tf.nn.relu(network)
    network = tf.layers.conv1d(inputs=network, filters=growthRate, kernel_size=kernel_size, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.concat((x, network), 2)
    return network


def Pool_block(x):
    network = tf.layers.batch_normalization(inputs=x)
    network = tf.nn.relu(network)
    network = tf.layers.average_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    return network


def head_cnn(x):
    network = tf.layers.conv1d(inputs=x, filters=64, kernel_size=50, strides=6,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 500
    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    # 250
    return network

def Dense_block(x,growthRate,kernel_size):
    network = Bottleneck(x, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    return network


def Dense_net(x,growthRate=12,kernel_size=8,keep_prob_=0.5):
    network = head_cnn(x)

    # 250
    network = Dense_block(network,growthRate, kernel_size)
    network = Pool_block(network)

    # 125
    network = Dense_block(network,growthRate, kernel_size)
    network = Pool_block(network)

    # 64
    network = Dense_block(network,growthRate, kernel_size)
    network = Pool_block(network)

    # 32
    network = Dense_block(network,growthRate, kernel_size)
    network = Pool_block(network)

    # 16
    network = Dense_block(network,growthRate, kernel_size)

    network = tf.layers.average_pooling1d(inputs=network, pool_size=16, strides=16, padding='same')
    return network
