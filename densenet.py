# -*- coding: utf-8 -*-
import tensorflow as tf
from cbam1d import se_block, cbam_block


def Bottleneck(x,growthRate,kernel_size):
    interChannels = 2 * growthRate
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
    # network = cbam_block(network)
    # network = se_block(network)
    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')
    network = tf.nn.dropout(network, keep_prob_)
    # 125
    return network


def Dense_block(x,growthRate,nDenseBlocks,kernel_size):
    for _ in range(int(nDenseBlocks)):
        x = Bottleneck(x, growthRate, kernel_size)
    # x = cbam_block(x)
    # x = se_block(x)
    return x


def Dense_net(x,growthRate=16,kernel_size=5,keep_prob_=0.8):
    growthRate = 16
    kernel_size = 5
    keep_prob_ = 1

    network = head_cnn(x,keep_prob_)

    # 125
    network = Dense_block(x=network,nDenseBlocks=2,growthRate=growthRate, kernel_size=kernel_size)
    out_cha = 96
    network = Pool_block(network,out_cha, keep_prob_)

    # 64
    network = Dense_block(x=network,nDenseBlocks=4,growthRate=growthRate, kernel_size=kernel_size)
    out_cha = 128
    network = Pool_block(network,out_cha, keep_prob_)

    # 32
    network = Dense_block(x=network,nDenseBlocks=6,growthRate=growthRate, kernel_size=kernel_size)
    out_cha = 192
    network = Pool_block(network,out_cha, keep_prob_)

    # 16

    network = Dense_block(x=network,nDenseBlocks=6,growthRate=growthRate, kernel_size=kernel_size)
    out_cha = 256
    network = Pool_block(network,out_cha, keep_prob_)

    # 8
    network = Dense_block(x=network,nDenseBlocks=4,growthRate=growthRate, kernel_size=kernel_size)

    # 1
    # network = cbam_block(network)
    # network = se_block(network)
    network = tf.layers.average_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')
    return network