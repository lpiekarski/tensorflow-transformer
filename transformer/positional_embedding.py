import tensorflow as tf
import numpy as np


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, maximum_position_encoding, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(maximum_position_encoding, d_model)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        return x
