import tensorflow as tf
from transformer.encoder_block import EncoderBlock


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, num_encoder_blocks, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder_blocks = []
        for _ in range(num_encoder_blocks):
            self.encoder_blocks.append(EncoderBlock(num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size))

    def __call__(self, x, *args, **kwargs):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
