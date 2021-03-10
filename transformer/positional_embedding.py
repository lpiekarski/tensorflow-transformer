import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, model_depth, mask_zero=True, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=model_depth, mask_zero=mask_zero)

    def __call__(self, x):
        return self.embedding(x)
