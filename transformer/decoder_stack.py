import tensorflow as tf
from transformer.decoder_block import DecoderBlock


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, num_decoder_blocks, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, model_depth, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.decoder_blocks = []
        for i in range(num_decoder_blocks):
            self.decoder_blocks.append(DecoderBlock(num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, model_depth, vocab_size, i == num_decoder_blocks - 1))

    def __call__(self, x, encoder_output, *args, **kwargs):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output)
        return x
