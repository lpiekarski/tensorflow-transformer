import tensorflow as tf

from transformer.multi_head_attention import MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.multi_head_self_attention = MultiHeadAttention(d_model, num_heads)
        self.multi_head_encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)

        self.ffn = self._point_wise_feed_forward_network(d_model, dff)

        self.layer_normalisation_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalisation_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalisation_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)

    def _point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        self_attention, self_attention_weights_block = self.multi_head_self_attention(x, x, x, look_ahead_mask)
        self_attention = self.dropout_1(self_attention, training=training)
        out1 = self.layer_normalisation_1(self_attention + x)

        encoder_decoder_attention, encoder_decoder_attention_weights_block = self.multi_head_encoder_decoder_attention(
            encoder_output,
            encoder_output,
            out1,
            padding_mask
        )
        encoder_decoder_attention = self.dropout_2(encoder_decoder_attention, training=training)
        out2 = self.layer_normalisation_2(encoder_decoder_attention + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_3(ffn_output, training=training)
        out3 = self.layer_normalisation_3(ffn_output + out2)

        return out3, self_attention_weights_block, encoder_decoder_attention_weights_block
