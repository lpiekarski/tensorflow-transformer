import tensorflow as tf
from transformer.decoder_stack import DecoderStack
from transformer.encoder_stack import EncoderStack
from transformer.positional_embedding import PositionalEmbedding
from tokenizer import Tokenizer
import numpy as np


class Transformer:
    def __init__(self, tokenizer: Tokenizer, model_depth, embedding_size, num_encoder_blocks, num_decoder_blocks, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, *args, **kwargs):
        vocab_size = tokenizer.get_num_tokens()

        inputs = tf.keras.layers.Input(shape=(model_depth,))
        decoder_inputs = tf.keras.layers.Input(shape=(model_depth,))
        decoder_alone_inputs = tf.keras.layers.Input(shape=(model_depth, embedding_size,))
        positional_embedding =PositionalEmbedding(vocab_size, embedding_size, model_depth)
        encoder_stack = EncoderStack(num_encoder_blocks, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size)
        decoder_stack = DecoderStack(num_decoder_blocks, num_heads, key_dim, dense_hidden_layers, dense_hidden_layer_size, embedding_size, model_depth, vocab_size)

        imputs_embedded = positional_embedding(inputs)
        decoder_inputs_embedded = positional_embedding(decoder_inputs)
        encoder_output = encoder_stack(imputs_embedded)
        decoder_output = decoder_stack(decoder_inputs_embedded, encoder_output)
        decoder_alone_output = decoder_stack(decoder_inputs_embedded, decoder_alone_inputs)

        self.encoder_decoder = tf.keras.models.Model(inputs=[inputs, decoder_inputs], outputs=decoder_output)
        self.encoder = tf.keras.models.Model(inputs=inputs, outputs=encoder_output)
        self.decoder = tf.keras.models.Model(inputs=[decoder_alone_inputs, decoder_inputs], outputs=decoder_alone_output)

        optimizer = tf.keras.optimizers.Adam()
        self.encoder_decoder.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
            'acc'
        ])

        self.model_depth = model_depth
        self.tokenizer = tokenizer

    def _count_len_lines(self, lines):
        ret = 0
        for line in lines:
            ret += self._count_len(line)
        return ret

    def _count_len(self, line):
        ret = 0
        for token in line:
            ret += len(self.tokenizer.tokens[token])
        return ret

    def _hot_encode(self, token):
        ret = np.zeros(self.tokenizer.get_num_tokens(), dtype='int32')
        ret[token] = 1
        return ret

    def fit(self, text, epochs=4, lr_hi=0.001, lr_low=0.00001, batch_size=128, validation_split=0.2):
        text_tokenized = self.tokenizer.tokenize(text)
        num_tokens = len(text_tokenized)
        inputs = []
        inputs2 = []
        outputs = []
        lines = []
        answer_line = None
        i = 0
        while i < num_tokens - 1:
            new_line = answer_line
            answer_line = []
            while i < num_tokens - 1 and not self.tokenizer.detokenize([text_tokenized[i]]).endswith('\n'):
                answer_line.append(text_tokenized[i])
                i += 1
            answer_line.append(text_tokenized[i])
            i += 1
            if new_line is None:
                continue
            lines.append(new_line)
            while self._count_len_lines(lines) > self.model_depth and len(lines) > 1:
                lines.pop(0)
            curr_input = []
            curr_output = []
            curr_output_he = []
            for line in lines:
                for token in line:
                    curr_input.append(token)
            for token in answer_line:
                inputs.append(curr_input)
                inputs2.append(curr_output.copy())
                curr_output.append(token)
                curr_output_he.append(self._hot_encode(token))
                outputs.append(curr_output_he[-1])
        outputs = np.array(outputs)
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding="post", maxlen=self.model_depth)
        padded_inputs2 = tf.keras.preprocessing.sequence.pad_sequences(inputs2, padding="post", maxlen=self.model_depth)

        def lr_schedule(ep, lr):
            r = float(ep) / epochs
            lr = r * lr_low + (1 - r) * lr_hi
            return lr

        self.encoder_decoder.fit(x=[padded_inputs, padded_inputs2], y=outputs, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_split, shuffle=True,
                                 callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

    def predict_next_line(self, text, max_tokens=256):
        text_tokenized = self.tokenizer.tokenize(text)
        if len(text_tokenized) > self.model_depth:
            text_tokenized = text_tokenized[-self.model_depth:]
        inputs = np.array([text_tokenized])
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding="post", maxlen=self.model_depth)
        inputs2 = []
        padded_inputs2 = np.zeros(padded_inputs.shape)
        encoder_output = self.encoder([padded_inputs])
        ret = ""
        num_tokens = 0
        while True:
            pia = np.array(padded_inputs2)
            result = self.decoder([encoder_output, pia])[0]
            inputs2.append(result)
            if len(inputs2) > self.model_depth:
                inputs2.pop(0)
            padded_inputs2 = tf.keras.preprocessing.sequence.pad_sequences(inputs2, padding="post", maxlen=self.model_depth)
            token = self.tokenizer.detokenize_hot_encoded(result)
            ret += token
            num_tokens += 1
            if token.endswith('\n') or num_tokens >= max_tokens:
                break
        return ret

    def get_distribution_of_next_token(self, text):
        text_tokenized = self.tokenizer.tokenize(text)
        if len(text_tokenized) > self.model_depth:
            text_tokenized = text_tokenized[-self.model_depth:]
        inputs = np.array([text_tokenized])
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding="post", maxlen=self.model_depth)
        padded_inputs2 = np.zeros(padded_inputs.shape)
        encoder_output = self.encoder([padded_inputs])
        result = self.decoder([encoder_output, padded_inputs2])[0]
        ret = []
        num_tokens = self.tokenizer.get_num_tokens()
        for i in range(num_tokens):
            ret.append({'token': self.tokenizer.tokens[i], 'probability': result[i].numpy()})
        ret.sort(key=lambda x: x['probability'], reverse=True)
        return ret
