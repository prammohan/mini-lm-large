import torch
import torch.nn as nn
import time
from tqdm import tqdm
from reader import read_file
from tokenizer import Tokenizer
from dataset import Dataset
from embedding import Embedding

class qkv:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # below has to be a matrix or nxn or nxm but 2D
        # why cant it be 1D?
        # because the input training sequence is 1D. 1Dx1D = 1D
        # and that means the inner dimension will always need to be equal
        # but that's not the only thing. The input token will always have element 3 (say)
        # match with element 3 of the weight matrix. Cannot talk to the other dims.
        # hence needs 2D to create relationships

        #0.02 is trying to reduce how big the numbers get when we do attention.
        #instead of 0.02 you could also use nn.Linear - this makes numbers somewhat smaller compared to randn
        #also nn.Parameter is within nn.Linear - so it is a built in fn to create parameters, register and init them
        self.W_q = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)
        #self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.W_k = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)
        self.W_v = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)

if __name__ == "__main__":
    start = time.time()
    x = qkv(64)
    print(f"Q, K, V initialized to: {x.W_q, x.W_k, x.W_v}")
    
    
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
    
    print (f"Embedding is {input_tensors.shape}")
    print (f"Wq is {x.W_q.shape}")
    print (f"Wk is {x.W_k.shape}")
    print (f"Wv is {x.W_v.shape}")

    batch_size = 32
    for i in range(0, len(input_tensors), batch_size):
        batch = input_tensors[i : i + batch_size]
        input_vectors = e(batch)
        q = input_vectors@x.W_q
        k = input_vectors@x.W_k
        v = input_vectors@x.W_v

        if i == 0:
            print(f"Batch shape: {batch.shape}")
            print (f"q is {q.shape}")
            print (f"k is {k.shape}")
            print (f"v is {v.shape}")
        
        scores = q @ k.transpose(-2, -1)
        if i == 0:
            print(f"After Q.Kt: {scores.shape}")

        # Scale — prevents huge numbers from making softmax saturate
        scores = scores / (embedding_dimension ** 0.5)

        # Mask — token 3 can't see tokens 4, 5, etc.
        # Tril - refers to lower triangle of the matrix 
        # Lower triangle are ones, upper are zeroes
        # So only lower triangle is considered (upper triangle scores are ignored)
        mask = torch.tril(torch.ones(sequence_length, sequence_length))

        # we use negative infinity as e^-inf=0 for all places where mask is zero (upper triangle)
        # if we zero e^0=1 which is still some probability
        # e^score happens in softmax (next)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Now softmax — turns scores into probabilities (0 to 1, rows sum to 1)
        # remeber scores are bs@seqlen@seqlen (3 dim)
        # And masks are temporal along the rows
        # So softmax across each row
        # Which means probabilities (pick a row) across all the columns - which is dim=-1
        # who is the most promising of all the elements in the columns (for a given row)
        # You can also say dim=2 but dim=-1 always gives the last dimension
        weights = torch.softmax(scores, dim=-1)

        outputs = weights @ v

        if i == ((len(input_tensors) // batch_size) * batch_size): #last batch
            print(f"Scores shape: {scores.shape}")
            
            # Print the 16x16 score grid for the few examples
            print(f"\nScores & Weights for example 11 - or the 11th input tensor:")
            print(scores[11])
            torch.set_printoptions(precision=50, sci_mode=False, linewidth=120)
            print(f"Weights shape: {weights.shape}")
            print(weights[11])
            
            print(f"Outputs shape: {outputs.shape}")
            print(outputs[11])
    
    
    del q, k, v, input_vectors


    end = time.time()
    print(f"Runtime is {end-start} seconds")