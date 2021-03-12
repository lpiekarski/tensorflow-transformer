import tensorflow as tf
from transformer.decoder_block import DecoderBlock


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_blocks = []
        for i in range(num_layers):
            self.decoder_blocks.append(DecoderBlock(d_model, num_heads, dff, dropout_rate))

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, self_attention_block, encoder_decoder_attention_block = self.decoder_blocks[i](x, encoder_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_self_attention'.format(i + 1)] = self_attention_block
            attention_weights['decoder_layer{}_encoder_decoder_attention'.format(i + 1)] = encoder_decoder_attention_block

        return x, attention_weights
