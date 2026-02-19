import torch
import torch.nn as nn
from reader import read_file
from tokenizer import Tokenizer
from dataset import Dataset

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, tokenid):
        return self.embedding(tokenid)
    
if __name__ == "__main__":
    #e = Embedding(65, 64)
    #print (f"Embedding is {e}")
    content = read_file("example.txt")
    t = Tokenizer(content)
    tokens = t.encode(content)
    text = t.decode(tokens)
    size = t.vocab_size(text)
    sequence_length = 16
    d = Dataset(tokens, sequence_length)
    training_set = d.make_ngrams()    
    training_examples = d.get_examples()
    input_shape, target_shape = d.get_shape(training_examples)
    print(f"Input shape is: {input_shape}")
    print(f"Target shape is: {target_shape}")
    inputs  = [x for x, y in training_examples]
    targets = [y for x, y in training_examples]
    embedding_dimension = 64
    e = Embedding(size, embedding_dimension)
    input_tensors = torch.tensor(inputs)
    input_vectors = e(input_tensors)
    print (f"Embedding is {input_vectors.shape}")
    for i in range(3):
        print(f"Token {i} (ID={input_tensors[0, i].item()}): {input_vectors[0, i][:8]}...")
    

