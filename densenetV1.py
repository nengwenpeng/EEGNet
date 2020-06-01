# -*- coding: utf-8 -*-
import tensorflow as tf


def Bottleneck(x,growthRate,kernel_size):
    interChannels = 4 * growthRate
    network = tf.layers.batch_normalization(inputs=x)
    network = tf.nn.relu(network)
    network = tf.layers.conv1d(inputs=network, filters=interChannels, kernel_size=1, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    network = tf.layers.conv1d(inputs=network, filters=growthRate, kernel_size=kernel_size, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.concat((x, network), 2)
    return network


def Pool_block(x,out_cha,keep_prob_=0.8):
    network = tf.layers.batch_normalization(inputs=x)
    network = tf.nn.relu(network)
    network = tf.layers.conv1d(inputs=network, filters=out_cha, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, use_bias=False)
    network = tf.layers.average_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    network = tf.nn.dropout(network, keep_prob_)
    return network

def head_cnn(x,keep_prob_=0.8):
    network = tf.layers.conv1d(inputs=x, filters=64, kernel_size=50, strides=6,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 500
    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')
    network = tf.nn.dropout(network, keep_prob_)
    # 125
    # 1
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)

    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    network = tf.nn.dropout(network, keep_prob_)
    # 64    (none  64 128)
    return network

def Dense_block(x,growthRate=12,kernel_size=3):
    network = Bottleneck(x, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    network = Bottleneck(network, growthRate, kernel_size)
    return network


def Dense_net(x,growthRate=24,kernel_size=3,keep_prob_=0.5):
    network = head_cnn(x, keep_prob_)
    # 64

    network = Dense_block(network, growthRate, kernel_size)
    out_cha = 128
    network = Pool_block(network,out_cha, keep_prob_)
    # 32

    network = Dense_block(network, growthRate, kernel_size)
    out_cha = 256
    network = Pool_block(network,out_cha,keep_prob_)
    # 16

    network = Dense_block(network, growthRate, kernel_size)
    out_cha = 512
    network = Pool_block(network,out_cha,keep_prob_)
    # 8
    network = Dense_block(network, growthRate, kernel_size)

    network = tf.layers.average_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')

    return network
