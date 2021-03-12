from tokenizer import Tokenizer
from transformer.frontend import NLP
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_merges = 40000
tokenization_path = f"tokenization_{num_merges}.json"
input_path = 'input.txt'

tokenizer = Tokenizer()
if os.path.exists(tokenization_path):
    tokenizer.load(tokenization_path)
else:
    tokenizer.from_file(input_path, num_merges)
    tokenizer.save(tokenization_path)

nlp = NLP(tokenizer, maximum_position_encoding=1000, d_model=64, num_layers=5, dff=1024, num_heads=8)
with open(input_path, 'r', encoding='utf-8') as f:
    nlp.train(f.read(), prev_tokens=128, epochs=10, evaluate_str=[
        'XD',
    ])
str = input()
nlp.generate_text(str, length=200)
#while True:
#    output, in_tokens, translated_tokens, attention_weights = nlp.evaluate(str)
    #print('input:', repr(str + "\n"))
    #print('output:', repr(output))
    #print('distribution:', nlp.distribution(str + "\n"))
#    print(repr(output))
#    str += output
    #nlp.plot_attention_weights(in_tokens, translated_tokens, attention_weights['decoder_layer4_encoder_decoder_attention'][0])
    #nlp.plot_attention_weights(in_tokens, translated_tokens, attention_weights['decoder_layer3_encoder_decoder_attention'][0])
    #nlp.plot_attention_weights(in_tokens, translated_tokens, attention_weights['decoder_layer2_encoder_decoder_attention'][0])
    #nlp.plot_attention_weights(in_tokens, translated_tokens, attention_weights['decoder_layer1_encoder_decoder_attention'][0])
print('')