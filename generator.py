import torch
from embedding import Embedding
from qkv import Attention, FFN, Head
from reader import read_file
from tokenizer import Tokenizer
import time


class Generator:
    def __init__(self, tokenizer, embedding, attention, ffn, head, seq_len):
        self.tokenizer = tokenizer
        self.e = embedding
        self.attn = attention
        self.ffn = ffn
        self.head = head
        self.seq_len = seq_len

    def generate(self, text, num_chars):
        tokens = self.tokenizer.encode(text)

        for _ in range(num_chars):
            input_ids = tokens[-self.seq_len:]
            input_tensor = torch.tensor([input_ids])

            input_vectors = self.e(input_tensor)
            output = self.attn(input_vectors)
            output = self.ffn(output)
            logits = self.head(output)

            next_token_logits = logits[0, -1, :]
            #next_token = torch.argmax(next_token_logits).item()
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

            # Print each character as it's generated
            ch = self.tokenizer.id_to_char[next_token]
            print(ch, end='', flush=True)
            time.sleep(0.03)

        print()
        return self.tokenizer.decode(tokens)
    
if __name__ == '__main__':
    content = read_file("example.txt")
    t = Tokenizer(content)
    e = Embedding(65, 64)
    attn = Attention(64)
    ffn = FFN(64, 256)
    head = Head(64, 65)

    checkpoint = torch.load('model_epoch20.pt')
    e.load_state_dict(checkpoint['embedding'])
    attn.load_state_dict(checkpoint['attention'])
    ffn.load_state_dict(checkpoint['ffn'])
    head.load_state_dict(checkpoint['head'])

    gen = Generator(t, e, attn, ffn, head, 16)
    print(gen.generate("First", 500))