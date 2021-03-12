import json
import numpy as np
from collections import defaultdict


class Tokenizer:
    def __init__(self):
        self.tokens = []
        self.reverse_lookup = None

    def load(self, filename):
        with open(filename, 'r') as f:
            self.tokens = json.loads(f.read())

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.tokens))

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for i in range(len(vocab) - 1):
            if '\n' in vocab[i + 1] or ' ' in vocab[i + 1] or '\t' in vocab[i + 1]:
                continue
            pairs[vocab[i], vocab[i + 1]] += 1
        return pairs

    def _merge_vocab(self, pair, v_in):
        v_out = []
        i = 0
        lvin = len(v_in)
        while i < lvin:
            if i < lvin - 1 and v_in[i] == pair[0] and v_in[i + 1] == pair[1]:
                v_out.append(v_in[i] + v_in[i + 1])
                i += 2
            else:
                v_out.append(v_in[i])
                i += 1

        return v_out

    def _get_tokens(self, vocab):
        ret = set()
        for word in vocab:
            ret.add(word)
        return list(ret)

    def initialize(self, base_text, num_merges=128, verbose=1):
        vocab = [str(letter) for letter in base_text]
        for i in range(num_merges):
            if verbose > 0:
                print(i + 1, '/', num_merges)
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] <= 1:
                break
            if verbose > 0:
                print('\tCreating token:', best)
            vocab = self._merge_vocab(best, vocab)
        self.tokens = self._get_tokens(vocab)
        self.tokens.sort(key=lambda x: len(x), reverse=True)
        self.tokens.insert(0, '<end>')
        self.tokens.insert(0, '<start>')
        self.tokens.insert(0, '<pad>')

    def from_file(self, filename, num_merges=128, verbose=1):
        with open(filename, 'r', encoding='utf-8') as f:
            base_text = f.read()
            self.initialize(base_text, num_merges, verbose)

    def get_tokens(self):
        return self.tokens

    def get_num_tokens(self):
        return len(self.tokens)

    def token_to_string(self, token):
        return self.tokens[token]

    def tokenize(self, text):
        maxlen = len(self.tokens[1])
        if self.reverse_lookup is None:
            self.reverse_lookup = {}
            q = 0
            for token in self.tokens:
                self.reverse_lookup[token] = q
                q += 1
        ret = []
        while len(text) > 0:
            if len(text) == 1:
                if text not in self.reverse_lookup:
                    ret.append(0)
                else:
                    ret.append(self.reverse_lookup[text])
                break
            i = maxlen
            prefix = text[0:i]
            while prefix not in self.reverse_lookup and i > 0:
                i -= 1
                prefix = text[0:i]
            if prefix not in self.reverse_lookup:
                v = 0
                text = text[1:]
            else:
                v = self.reverse_lookup[prefix]
                text = text[len(prefix):]
            ret.append(v)
        return np.array(ret)

    def detokenize_hot_encoded(self, token_list):
        ret = ""
        for token_hot_encoded in token_list:
            ret = ret + self.tokens[np.argmax(token_hot_encoded)]
        return ret

    def detokenize(self, token_list):
        ret = ""
        for token_id in token_list:
            if token_id == 0 or token_id == 1 or token_id == 2:
                continue
            ret = ret + self.tokens[token_id]
        return ret

    def print_info(self, text):
        ret = ''
        tokenized = self.tokenize(text)
        for token in tokenized:
            detokenized = self.detokenize([token])
            ret += str(repr(detokenized)) + ' ' + str(token) + ', '
        return ret