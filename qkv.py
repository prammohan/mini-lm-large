import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
from tqdm import tqdm
from reader import read_file
from tokenizer import Tokenizer
from dataset import Dataset
from embedding import Embedding
from torchviz import make_dot
from torch.profiler import profile, ProfilerActivity, record_function

class QKV(nn.Module):
    def __init__(self, embed_dim):
        # super is to init the parent nn class
        super().__init__()
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
        #self.W_q = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)
        #self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.W_q = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)
        self.W_k = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)
        self.W_v = nn.Parameter( torch.randn(self.embed_dim, self.embed_dim) * 0.02)

    def forward(self, x):
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        return q, k, v

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv = QKV(embed_dim)

    def forward(self, x):
        q, k, v = self.qkv(x)
        scores = q @ k.transpose(-2, -1)

        # Scale — prevents huge numbers from making softmax saturate
        scores = scores / (self.embed_dim ** 0.5)
        seq_len = x.shape[-2]

        # Mask — token 3 can't see tokens 4, 5, etc.
        # Tril - refers to lower triangle of the matrix 
        # Lower triangle are ones, upper are zeroes
        # So only lower triangle is considered (upper triangle scores are ignored)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

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
        output = weights @ v
        return output

class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
class Head(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim, vocab_size)*0.02)

    def forward(self, x):
        return x @ self.W


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()
    embedding_dimension = 64
    attn = Attention(embedding_dimension)
    ffn = FFN(embedding_dimension, hidden_dim=256)

    #print(f"Q, K, V initialized to: {attn.qkv.W_q, attn.qkv.W_k, attn.qkv.W_v}")
    
    
    content = read_file("example.txt")
    t = Tokenizer(content)
    tokens = t.encode(content)
    text = t.decode(tokens)
    size = t.vocab_size(text)

    head = Head(embedding_dimension, size)

    sequence_length = 16
    d = Dataset(tokens, sequence_length)
    loader = DataLoader(d, batch_size=32, shuffle=True)
    
    
    embedding_dimension = 64
    e = Embedding(size, embedding_dimension)
    # Collect all learnable parameters
    params = (
        list(e.parameters()) +
        list(attn.parameters()) +
        list(ffn.parameters()) +
        list(head.parameters())
    )
    
    optimizer = torch.optim.Adam(params, lr=0.001)

    epochs = 50
    batch_size = 32
    do_profile = False
    first_iteration = True;

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        
        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            targets_batch = batch_y.to(device)
            
            # Forward Pass
            with record_function("embedding"):
                input_vectors = e(batch_x).to(device)
            
            #if (first_iteration == True):
                #print(f"batchx shape is {batch_x.shape}")
                #print(f"Number of examples: {len(batch_x)}")
                #print(f"Each input length:  {len(batch_x[0])}")
                #print(f"Each target length: {len(batch_x[0])}")
                #print (f"Embedding is {input_vectors.shape}")
                #print (f"Wq is {attn.qkv.W_q.shape}")
                #print (f"Wk is {attn.qkv.W_k.shape}")
                #print (f"Wv is {attn.qkv.W_v.shape}")
            
            first_iteration = False;
    
            if epoch == 0 and first_iteration == True:
                do_profile = True
           
            if do_profile:
                prof = profile(activities=[ProfilerActivity.CPU])
                prof.__enter__()
            
            with record_function("attention"):
                attn_outputs = attn(input_vectors).to(device)
            with record_function("ffn"):
                ffn_outputs = ffn(attn_outputs).to(device)
            with record_function("head"):
                logits = head(ffn_outputs).to(device)

            logits_flat = logits.view(-1, size) #keeps the vocab size but collapses the batch and seq len dimension
            targets_flat = targets_batch.view(-1) #(bs, seq_len)
            
            #Compute Loss
            with record_function("loss"):
                loss = F.cross_entropy(logits_flat, targets_flat)
            #print(loss.item())
            
            # Backward Pass
            with record_function("optimizer_zero_grad"):
                optimizer.zero_grad()   # clear old gradients
            
            with record_function("backward"):
                loss.backward() # new gradients
            
            with record_function("optimizer_step"):
                optimizer.step()        # update weights

            if do_profile:
                prof.__exit__(None, None, None)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
                prof.export_chrome_trace("trace.json")
                do_profile = False

            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    end = time.time()
    print(f"Runtime is {end-start} seconds")
    print(e)
    print(attn)
    print(ffn)
    print(head)
    make_dot(loss, params=dict(list(attn.named_parameters()) + 
                        list(ffn.named_parameters()) + 
                        list(head.named_parameters()) + 
                        list(e.named_parameters()))).render("model_graph", format="png")
    
    # Save the model
    torch.save({
    'embedding': e.state_dict(),
    'attention': attn.state_dict(),
    'ffn': ffn.state_dict(),
    'head': head.state_dict(),
    }, f'model_epoch{epoch+1}.pt')
    
