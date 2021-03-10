import tensorflow as tf


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_self_attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.add_1 = tf.keras.layers.Add()
        self.layer_normalization_1 = tf.keras.layers.LayerNormalization()
        self.dense_hidden_layers = []
        for _ in range(dense_hidden_layers):
            self.dense_hidden_layers.append(tf.keras.layers.Dense(dense_hidden_layer_size, activation='relu'))
        self.last_dense_layer = tf.keras.layers.Dense(embedding_size)
        self.add_2 = tf.keras.layers.Add()
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization()

    def __call__(self, x):
        res = x
        x = self.multi_head_self_attention(x, x)
        x = self.add_1([x, res])
        x = self.layer_normalization_1(x)
        res = x
        for dense_hidden_layer in self.dense_hidden_layers:
            x = dense_hidden_layer(x)
        x = self.last_dense_layer(x)
        x = self.add_2([x, res])
        x = self.layer_normalization_2(x)
        return x
