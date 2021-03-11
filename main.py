from tokenizer import Tokenizer
from transformer.frontend import NLP
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_merges = 1024
tokenization_path = f"tokenization_{num_merges}.json"
input_path = 'input.txt'
model_depth = 8
embedding_size = 8
num_encoder_blocks = 12
num_decoder_blocks = 12
num_heads = 2
key_dim = 8

tokenizer = Tokenizer()
if os.path.exists(tokenization_path):
    tokenizer.load(tokenization_path)
else:
    tokenizer.from_file(input_path, num_merges)
    tokenizer.save(tokenization_path)

nlp = NLP(tokenizer, maximum_position_encoding=1000, d_model=64, num_layers=6, dff=128, num_heads=4)

print(nlp.evaluate("Pan Tadeusz\n")[0])

with open(input_path, 'r', encoding='utf-8') as f:
    nlp.train(f.read(), previous_lines_max_length=64)

print(nlp.evaluate("Pan Tadeusz\n")[0])
print('')
