import tensorflow as tf
from transformer.decoder_stack import DecoderStack
from transformer.encoder_stack import EncoderStack
from transformer.positional_embedding import PositionalEmbedding


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.positional_embed = PositionalEmbedding(d_model, vocab_size, maximum_position_encoding)
        self.tokenizer = EncoderStack(num_layers, d_model, num_heads, dff, dropout_rate)
        self.decoder = DecoderStack(num_layers, d_model, num_heads, dff, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)  # softmax?

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp = self.positional_embed(inp)
        tar = self.positional_embed(tar)
        enc_output = self.tokenizer(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
