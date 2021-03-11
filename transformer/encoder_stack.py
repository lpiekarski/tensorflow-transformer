import tensorflow as tf
from transformer.encoder_block import EncoderBlock


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_blocks = []
        for _ in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(d_model, num_heads, dff, dropout_rate))

    def call(self, x, training, mask):
        x = self.dropout(x, training=training)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training, mask)

        return x

