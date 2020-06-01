# -*- coding: utf-8 -*-
import tensorflow as tf


def flatten(name, input_var):
    # 对数据进行 压扁暂时不知道输入维度形式，把卷积以后的 压缩成向量
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                            name=name)

    return output_var


def Dense_net(x,growthRate=24,kernel_size=3,keep_prob_=0.5):
    # List to store the output of each CNNs
    output_conns = []
    ######### CNNs with small filter size at the first layer #########

    # Convolution   y = f(x) 这种写法，特别相似,比较像直接写前向传递函数，而不是先定义
    network = tf.layers.conv1d(inputs=x, filters=64, kernel_size=50, strides=6,
                               padding='same', activation=tf.nn.relu)
    # network = tf.layers.conv1d(inputs=input_var, filters=64, kernel_size=50, strides=6,

    network = tf.layers.max_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    # Convolution
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)

    # Max pooling
    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')

    # Flatten
    network = flatten(name="flat1", input_var=network)

    output_conns.append(network)

    ######### CNNs with large filter size at the first layer #########

    # Convolution
    network = tf.layers.conv1d(inputs=x, filters=64, kernel_size=400, strides=50,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    # Convolution
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)

    # Max pooling
    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')

    # Flatten
    network = flatten(name="flat2", input_var=network)

    output_conns.append(network)

    # Concat
    network = tf.concat(output_conns, 1, name="concat1")

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    return network