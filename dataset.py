import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from reader import read_file, read_wikitext
from tokenizer import Tokenizer

class Dataset(TorchDataset):
    def make_ngrams(self):
        ngrams = []
        for i in range(len(self.token_ids) - self.seq_len + 1):
            ngrams.append(self.token_ids[i:i + self.seq_len])
        return ngrams
    
    def __init__(self, token_ids, seq_len: int = 64):
        #self.token_ids = token_ids
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def get_examples(self) -> list[tuple[list[int], list[int]]]:
    #    """
    #    Slide a window of size seq_len across token_ids.
    #    For each position, create:
    #      input  = token_ids[i : i + seq_len]
    #      target = token_ids[i+1 : i + seq_len + 1]  (shifted by 1)

    #    Returns a list of (input, target) tuples.
    #    """
        examples = []

        for i in range(len(self.token_ids) - self.seq_len):
            x = self.token_ids[i : i + self.seq_len]
            y = self.token_ids[i + 1 : i + self.seq_len + 1]
            examples.append((x, y))

        return examples
    
    def __getitem__(self, idx):
        # Because we cannot get all items at once - needed 44GB or something of memory
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return x, y
    
    def __len__(self):
        return len(self.token_ids) - self.seq_len
     
if __name__ == "__main__":
    content = read_wikitext()
    t = Tokenizer(content)
    tokens = t.encode(content)
    print(f"token is : {tokens[0]}")
    text = t.decode(tokens)
    size = t.vocab_size(text)
    print ("Vocab size is", size)
    print (f"Total tokens in dataset: {len(tokens):,}")
    sequence_length = 16
    d = Dataset(tokens, sequence_length)
    loader = DataLoader(d, batch_size=32, shuffle=True)
    
    first_iteration = True;
    for batch_x, batch_y in loader:
        training_examples = batch_x
        if (first_iteration == True):
            #print (f"Few inputs and target sets are: {training_examples[:10]}")
            print(f"Number of examples: {len(training_examples)}")
            print(f"Each input length:  {len(training_examples[0])}")
            print(f"Each target length: {len(training_examples[0])}")

        first_iteration = False;