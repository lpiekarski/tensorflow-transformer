import random
import time
import tensorflow as tf
from transformer.custom_schedule import CustomSchedule
from transformer.decoder_stack import DecoderStack
from transformer.encoder_stack import EncoderStack
from transformer.positional_embedding import PositionalEmbedding
from tokenizer import Tokenizer
import numpy as np
from transformer.transformer import Transformer
import matplotlib.pyplot as plt


class NLP:
    def __init__(self, tokenizer: Tokenizer, maximum_position_encoding=1000, num_layers=6, d_model=512, num_heads=8, dff=2048, dropout_rate=0.1):
        self.tokenizer = tokenizer
        self.d_model = d_model
        vocab_size = tokenizer.get_num_tokens()

        self.learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.transformer = Transformer(num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate)

        self.checkpoint_path = './checkpoints/train'

        self.ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def _create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def _create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def _create_masks(self, inp, tar):
        enc_padding_mask = self._create_padding_mask(inp)
        dec_padding_mask = self._create_padding_mask(inp)
        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self._create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def train(self, text, epochs=20, batch_size=64, prev_tokens=512, evaluate_str=None):
        start_token = 1
        end_token = 2
        text_tokenized = self.tokenizer.tokenize(text)
        train_x = []
        train_y = []
        one_token_learning = True
        if not one_token_learning:
            previous_lines = []
            current_line = []
            marker_token = self.tokenizer.tokenize("\n")[0]
            for token_id in text_tokenized:
                if token_id != marker_token:
                    current_line.append(token_id)
                else:
                    current_line.append(token_id)
                    if len(previous_lines) > 0:
                        while len(previous_lines) > prev_tokens:
                            previous_lines.pop(0)
                        train_x.append([start_token] + previous_lines.copy() + [end_token])
                        train_y.append([start_token] + current_line.copy() + [end_token])
                    previous_lines = previous_lines + current_line.copy()
                    current_line = []
        else:
            rolling_window = []
            for token_id in text_tokenized:
                if len(rolling_window) > 0:
                    train_x.append([start_token] + rolling_window.copy() + [end_token])
                    train_y.append([start_token, token_id, end_token])
                rolling_window.append(token_id)
                if len(rolling_window) > prev_tokens:
                    rolling_window.pop(0)

        def make_batches(ds):
            batches = [[0, ([], [])]]
            i = 0
            batch_id = 0
            batch_i = batch_size
            idx = list(range(0, len(ds[0])))
            random.shuffle(idx)
            while i < len(ds[0]):
                batches[batch_id][1][0].append(ds[0][idx[i]])
                batches[batch_id][1][1].append(ds[1][idx[i]])
                i += 1
                batch_i -= 1
                if batch_i == 0:
                    batch_id += 1
                    batch_i = batch_size
                    batches.append([batch_id, ([], [])])
            return batches

        train_batches = make_batches([train_x, train_y])

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        def accuracy_function(real, pred):
            accuracies = tf.equal(real, tf.argmax(pred, axis=2))
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            accuracies = tf.math.logical_and(mask, accuracies)
            accuracies = tf.cast(accuracies, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(accuracy_function(tar_real, predictions))

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in train_batches:
                inp = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(inp, padding="post"), dtype=tf.int64)
                tar = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(tar, padding="post"), dtype=tf.int64)
                train_step(inp, tar)

                if batch % 1 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            if (epoch + 1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
            if evaluate_str is not None:
                print('Evaluation:')
                for string in evaluate_str:
                    output, _, _, _ = self.evaluate(string)
                    print('\tinput:', repr(string))
                    print('\toutput:', repr(output))
                    print('\tdistribution:', self.distribution(string))
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    def distribution(self, text):
        start = [1]
        end = [2]
        text = np.array(start + self.tokenizer.tokenize(text).tolist() + end)
        text_padded = [text]

        encoder_input = tf.convert_to_tensor(text_padded)

        output = [start]
        output_padded = tf.convert_to_tensor(output)

        enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(encoder_input, output_padded)
        predictions, attention_weights = self.transformer(encoder_input, output_padded, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        ret = []
        for token_id in range(predictions.shape[2]):
            ret.append({
                'token': self.tokenizer.detokenize([token_id]),
                'score': predictions[0][0][token_id].numpy()
            })
        ret.sort(key=lambda x: x['score'], reverse=True)
        return ret

    def evaluate(self, text, max_length=40):
        start = [1]
        end = [2]
        text = np.array(start + self.tokenizer.tokenize(text).tolist() + end)
        text_padded = [text]

        encoder_input = tf.convert_to_tensor(text_padded)

        output = [start]
        output_padded = tf.convert_to_tensor(output)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(encoder_input, output_padded)
            predictions, attention_weights = self.transformer(encoder_input, output_padded, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output = tf.concat([output, predicted_id], axis=-1)
            output_padded = tf.convert_to_tensor(output)
            if predicted_id == end:
                break
        text = self.tokenizer.detokenize(output[0])
        in_tokens = []
        translated_tokens = []
        for token_id in encoder_input[0]:
            in_tokens.append(self.tokenizer.detokenize([token_id]))
        for token_id in output[0]:
            translated_tokens.append(self.tokenizer.detokenize([token_id]))
        return text, in_tokens, translated_tokens, attention_weights

    def plot_attention_head(self, in_tokens, translated_tokens, attention):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        translated_tokens = translated_tokens[1:]

        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = [repr(token) for token in in_tokens]
        ax.set_xticklabels(
          labels, rotation=90)

        labels = [repr(token) for token in translated_tokens]
        ax.set_yticklabels(labels)

    def plot_attention_weights(self, in_tokens, translated_tokens, attention_heads):
        fig = plt.figure(figsize=(16, 8))

        for h, head in enumerate(attention_heads):
            ax = fig.add_subplot(2, 4, h + 1)
            self.plot_attention_head(in_tokens, translated_tokens, head)
            ax.set_xlabel('Head {}'.format(h + 1))

        plt.tight_layout()
        plt.show()

    def generate_text(self, text, length=1000, verbose=0):
        while len(text) < length:
            eval_out, _, _, _ = self.evaluate(text)
            text += eval_out
        return text
