import tensorflow as tf


def Bi_GRU(x, output_dim,attention_size=32):
    gruCell_fw = tf.nn.rnn_cell.GRUCell(output_dim)
    gruCell_bw = tf.nn.rnn_cell.GRUCell(output_dim)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        output_dim,x, memory_sequence_length=None)

    gruCell_fw = tf.contrib.seq2seq.AttentionWrapper(
        gruCell_fw, attention_mechanism,
        attention_layer_size=attention_size,
        alignment_history=True)
    gruCell_bw = tf.contrib.seq2seq.AttentionWrapper(
        gruCell_bw, attention_mechanism,
        attention_layer_size=attention_size,
        alignment_history=True)

    (gru_fw_out, gru_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=gruCell_fw,
        cell_bw=gruCell_bw,
        inputs=x,
        dtype=tf.float32)
    outputs = tf.concat((gru_fw_out, gru_bw_out), 2)

    return outputs


def Dense_GRU(x, output_dim=64):
    with tf.variable_scope('gru_1'):
        output = Bi_GRU(x, output_dim)
    with tf.variable_scope('gru_2'):
        outputs = tf.concat((output, x), 2)
        outputs = Bi_GRU(outputs, output_dim)
    return outputs[:, -1, :]