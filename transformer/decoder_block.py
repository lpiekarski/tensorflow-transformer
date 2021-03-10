import tensorflow as tf


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, model_depth, vocab_size, last, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_self_attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((model_depth, model_depth)), -1, 0)
        self.add_1 = tf.keras.layers.Add()
        self.layer_normalisation_1 = tf.keras.layers.LayerNormalization()
        self.multi_head_encoder_decoder_attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.add_2 = tf.keras.layers.Add()
        self.layer_normalisation_2 = tf.keras.layers.LayerNormalization()
        self.dense_hidden_layers = []
        for _ in range(dense_hidden_layers):
            self.dense_hidden_layers.append(tf.keras.layers.Dense(dense_hidden_layer_size, activation='relu'))
        self.last_dense_layer = tf.keras.layers.Dense(embedding_size)
        self.add_3 = tf.keras.layers.Add()
        self.layer_normalisation_3 = tf.keras.layers.LayerNormalization()
        self.last = last
        if last:
            self.flatten = tf.keras.layers.Flatten()
            self.softmax_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def __call__(self, x, encoder_output):
        res = x
        x = self.multi_head_self_attention(x, x, attention_mask=self.look_ahead_mask)
        x = self.add_1([x, res])
        x = self.layer_normalisation_1(x)
        res = x
        x = self.multi_head_encoder_decoder_attention(x, encoder_output)
        x = self.add_2([x, res])
        x = self.layer_normalisation_2(x)
        res = x
        for dense_hidden_layer in self.dense_hidden_layers:
            x = dense_hidden_layer(x)
        x = self.last_dense_layer(x)
        x = self.add_3([x, res])
        x = self.layer_normalisation_3(x)
        if self.last:
            x = self.flatten(x)
            x = self.softmax_dense(x)
        return x
