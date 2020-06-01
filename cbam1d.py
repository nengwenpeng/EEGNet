import tensorflow as tf


def se_block(residual, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel = residual.get_shape()[-1]
    # Global average pooling
    squeeze = tf.reduce_mean(residual, axis=[1], keepdims=True)
    assert squeeze.get_shape()[1:] == (1, channel)
    excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel // ratio,
                                 activation=tf.nn.relu
                                 )
    assert excitation.get_shape()[1:] == (1, channel // ratio)
    excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid
                                 )
    assert excitation.get_shape()[1:] == (1, channel)
    scale = residual * excitation
    return scale


def cbam_block(input_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    attention_feature = channel_attention(input_feature, ratio)
    attention_feature = spatial_attention(attention_feature)
    return attention_feature


def channel_attention(input_feature, ratio=8):
    channel = input_feature.get_shape()[-1]
    avg_pool = tf.reduce_mean(input_feature, axis=[1], keepdims=True)

    assert avg_pool.get_shape()[1:] == (1, channel)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                               units=channel // ratio,
                               activation=tf.nn.relu
                               )
    assert avg_pool.get_shape()[1:] == (1, channel // ratio)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                               units=channel,
                               )
    assert avg_pool.get_shape()[1:] == (1, channel)

    max_pool = tf.reduce_max(input_feature, axis=[1], keepdims=True)
    assert max_pool.get_shape()[1:] == (1, channel)
    max_pool = tf.layers.dense(inputs=max_pool,
                               units=channel // ratio,
                               activation=tf.nn.relu
                               )

    assert max_pool.get_shape()[1:] == (1, channel // ratio)
    max_pool = tf.layers.dense(inputs=max_pool,
                               units=channel
                               )
    assert max_pool.get_shape()[1:] == (1, channel)

    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature):
    kernel_size = 13

    avg_pool = tf.reduce_mean(input_feature, axis=[2], keepdims=True)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(input_feature, axis=[2], keepdims=True)
    assert max_pool.get_shape()[-1] == 1
    concat = tf.concat([avg_pool, max_pool], 2)
    assert concat.get_shape()[-1] == 2

    concat = tf.layers.conv1d(concat,
                              filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding="same",
                              activation=None,
                              use_bias=False)
    assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat