import tensorflow as tf
from transformer.multi_head_attention import MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = self._point_wise_feed_forward_network(d_model, dff)

        self.layer_normalization_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    def _point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x, training, mask):
        self_attention, _ = self.multi_head_self_attention(x, x, x, mask)
        self_attention = self.dropout_1(self_attention, training=training)
        out1 = self.layer_normalization_1(x + self_attention)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out2 = self.layer_normalization_2(out1 + ffn_output)

        return out2
