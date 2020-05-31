import tensorflow as tf


def Bottleneck(x,growthRate,kernel_size):
    interChannels = 4 * growthRate
    # network = tf.layers.batch_normalization(inputs=x)
    # network = tf.nn.relu(network)
    network = tf.nn.relu(x)
    network = tf.layers.conv1d(inputs=network, filters=interChannels, kernel_size=1, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=x)
    network = tf.nn.relu(network)
    network = tf.layers.conv1d(inputs=network, filters=growthRate, kernel_size=kernel_size, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.concat((x, network), 2)
    return network


def Pool_block(x,out_cha,keep_prob_=0.8):
    # network = tf.layers.batch_normalization(inputs=x)
    # network = tf.nn.relu(network)
    network = tf.nn.relu(x)
    network = tf.layers.conv1d(inputs=network, filters=out_cha, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, use_bias=False)
    network = tf.layers.average_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    network = tf.nn.dropout(network, keep_prob_)
    return network

def head_cnn(x):
    network = tf.layers.conv1d(inputs=x, filters=64, kernel_size=50, strides=6,
                               padding='same', activation=None, use_bias=False)
    # network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 500
    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    # 250
    # 1
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=None, use_bias=False)
    # network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 2
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=None, use_bias=False)
    # network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 3
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=None, use_bias=False)
    network = tf.layers.batch_normalization(inputs=network)
    network = tf.nn.relu(network)
    # 250
    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')
    # 125    (none  125 128)

    return network

def Dense_block(x,in_cha,growthRate=12,kernel_size=3):
    network = Bottleneck(x, growthRate, kernel_size)
    network = Bottleneck(network, growthRate*1+in_cha, kernel_size)
    network = Bottleneck(network, growthRate*2+in_cha, kernel_size)
    network = Bottleneck(network, growthRate*3+in_cha, kernel_size)
    return network


def Dense_net(x,growthRate=12,kernel_size=3,keep_prob_=0.8):
    network = head_cnn(x)
    # 125
    in_cha = 128
    network = Dense_block(network, in_cha, growthRate, kernel_size)
    out_cha = 128
    network = Pool_block(network,out_cha,keep_prob_)
    # 62
    in_cha = out_cha
    network = Dense_block(network, in_cha, growthRate, kernel_size)
    out_cha = 256
    network = Pool_block(network,out_cha,keep_prob_)
    # 31
    in_cha = out_cha
    network = Dense_block(network, in_cha, growthRate, kernel_size)
    out_cha = 384
    network = Pool_block(network,out_cha,keep_prob_)
    # 15
    in_cha = out_cha
    network = Dense_block(network, in_cha, growthRate, kernel_size)
    out_cha = 512
    network = Pool_block(network,out_cha,keep_prob_)
    # 7
    network = Dense_block(network, in_cha, growthRate, kernel_size)

    network = tf.layers.average_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')

    return network



def flatten(input_var):
    # 对数据进行 压扁暂时不知道输入维度形式，把卷积以后的 压缩成向量
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                           )

    return output_var

#
# import numpy as np
#
# x_data = np.random.rand(10,3000,1)
# y_data = np.random.rand(10,5)
#
# xs = tf.placeholder(tf.float32, [None,3000,1])
# ys = tf.placeholder(tf.float32, [None, 5])
#
# y = Dense_net(xs)
# y = flatten(y)
# y = tf.layers.dense(y,5)
#
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=y)
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))