from tokenizer import Tokenizer
from transformer.transformer import Transformer
import os

num_merges = 1024
tokenization_path = f"tokenization_{num_merges}.json"
input_path = 'input.txt'
model_depth = 4
embedding_size = 8
num_encoder_blocks = 8
num_decoder_blocks = 8
num_heads = 4
key_dim = 4
dense_hidden_layers = 2
dense_hidden_layer_size = 16

tokenizer = Tokenizer()
if os.path.exists(tokenization_path):
    tokenizer.load(tokenization_path)
else:
    tokenizer.from_file(input_path, num_merges)
    tokenizer.save(tokenization_path)

transformer = Transformer(
    tokenizer,
    model_depth,
    embedding_size,
    num_encoder_blocks,
    num_decoder_blocks,
    num_heads,
    key_dim,
    dense_hidden_layers,
    dense_hidden_layer_size
)

with open(input_path, 'r', encoding='utf-8') as f:
    transformer.fit(f.read(), epochs=1)
print('')