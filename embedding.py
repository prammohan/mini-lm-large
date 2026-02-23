import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from reader import read_file, read_wikitext
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
    content = read_wikitext()
    t = Tokenizer(content)
    tokens = t.encode(content)
    text = t.decode(tokens)
    size = t.vocab_size(text)
    sequence_length = 16
    d = Dataset(tokens, sequence_length)
    loader = DataLoader(d, batch_size=32, shuffle=True)
    
    first_iteration = True;
    embedding_dimension = 64
    e = Embedding(size, embedding_dimension)
    for batch_x, batch_y in loader:
        input_vectors = e(batch_x)
        
        if (first_iteration == True):
            print(f"batchx shape is {batch_x.shape}")
            print(f"Number of examples: {len(batch_x)}")
            print(f"Each input length:  {len(batch_x[0])}")
            print(f"Each target length: {len(batch_x[0])}")
            print (f"Embedding is {input_vectors.shape}")
            for i in range(3):
                print(f"Token {i} (ID={input_tensors[0, i].item()}): {input_vectors[0, i][:8]}...")
   
        first_iteration = False;