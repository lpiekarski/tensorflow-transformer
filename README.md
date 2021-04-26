# Transformer model implementation in Tensorflow
This is a transformer model for natural language understanding. The model predicts a continuation of a given text.

## How to use
1. First off import these classes:
```python
from tokenizer import Tokenizer
from transformer.frontend import NLP
```
2. Tokenize the text you want the model to train on:
  ```python
  tokenizer = Tokenizer()
  tokenizer.from_file("train.txt", 5000)
  tokenizer.save("tokenization.json")
  # Later you can use this instead of tokenizing again from the start:
  # tokenizer.load("tokenization.json")
  ```
3. Initialize the model:
  ```python
  nlp = NLP(tokenizer, maximum_position_encoding=1000, d_model=64, num_layers=5, dff=1024, num_heads=8)
  ```
4. Train the model:
  ```python
  with open("train.txt", 'r', encoding='utf-8') as f:
      nlp.train(f.read(), prev_tokens=128, epochs=10, batch_size=64)
  ```
5. Get output from the model:
  ```python
  nlp.distribution("complete this")
  nlp.sample_top_k("complete this")
  nlp.generate_text("complete this", length=500)
  ```
