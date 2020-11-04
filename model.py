import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1),
                       tf.cast(x, dtype=tf.float32)], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


if __name__ == "__main__":

    sequnce_length = 100
    sequnce_dim = 164
    attention_dim = 256
    label_length = 19
    BATCH_SIZE = 32
    units = 1024

    # 模拟样本输入输出
    sample_input = np.random.rand(BATCH_SIZE, sequnce_length, sequnce_dim)
    next_input = np.random.rand(BATCH_SIZE, 1, sequnce_dim)

    # encoder
    encoder = Encoder(units, BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(sample_input, sample_hidden)

    # attention
    attention_layer = BahdanauAttention(attention_dim)
    attention_result, attention_weights = attention_layer(
        sample_hidden, sample_output)

    # decoder
    decoder = Decoder(label_length, units, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(next_input,
                                          sample_hidden, sample_output)

    # print(sample_decoder_output.shape)
