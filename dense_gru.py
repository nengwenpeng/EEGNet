import tensorflow as tf


def Bi_GRU(x, output_dim, attention_size, gru_layers):
    def gru_cell():
        gru = tf.contrib.rnn.GRUCell(output_dim)
        return gru

    gruCell_fw = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(gru_layers)])
    gruCell_bw = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(gru_layers)])

    attention_mechanism_fw = tf.contrib.seq2seq.LuongAttention(
        output_dim, x, memory_sequence_length=None)
    attention_mechanism_bw = tf.contrib.seq2seq.LuongAttention(
        output_dim, x, memory_sequence_length=None)

    gruCell_fw = tf.contrib.seq2seq.AttentionWrapper(
        gruCell_fw, attention_mechanism_fw,
        attention_layer_size=attention_size,
        alignment_history=True)

    gruCell_bw = tf.contrib.seq2seq.AttentionWrapper(
        gruCell_bw, attention_mechanism_bw,
        attention_layer_size=attention_size,
        alignment_history=True)

    (gru_fw_out, gru_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=gruCell_fw,
        cell_bw=gruCell_bw,
        inputs=x,
        dtype=tf.float32)

    outputs = tf.concat((gru_fw_out, gru_bw_out), 2)
    return outputs


def Dense_GRU(x, output_dim=64, attention_size=32, gru_layers=1):
    stfts = tf.contrib.signal.stft(x,
                                   frame_length=200,
                                   frame_step=100,
                                   fft_length=256)
    stfts = tf.abs(stfts)  # (?, 1, 29, 129)
    shape = stfts.get_shape().as_list()
    stfts = tf.reshape(stfts, [-1, shape[2], shape[3]])  # B,29,129

    with tf.variable_scope('gru_1'):
        outputs = Bi_GRU(stfts, output_dim, attention_size, gru_layers)
    with tf.variable_scope('gru_2'):
        outputs = tf.concat((outputs, stfts), 2)
        outputs = Bi_GRU(outputs, output_dim, attention_size, gru_layers)
    return outputs[:, -1, :]
