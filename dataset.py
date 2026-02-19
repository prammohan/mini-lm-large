import torch
from reader import read_file
from tokenizer import Tokenizer

class Dataset:
    def make_ngrams(self):
        ngrams = []
        for i in range(len(self.token_ids) - self.seq_len + 1):
            ngrams.append(self.token_ids[i:i + self.seq_len])
        return ngrams
    
    def __init__(self, token_ids, seq_len: int = 64):
        self.token_ids = token_ids
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
    
    def get_shape(self, examples):
        inputs  = [x for x, y in examples]
        targets = [y for x, y in examples]

        inputs_tensor  = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)
        return inputs_tensor.shape, targets_tensor.shape
    
if __name__ == "__main__":
    content = read_file("example.txt")
    t = Tokenizer()
    tokens = t.encode(content)
    print(f"token is : {tokens[0]}")
    text = t.decode(tokens)
    size = t.vocab_size(text)
    #print (tokens)
    #print (text)
    print ("Vocab size is", size)
    print (f"Total tokens in dataset: {len(tokens):,}")
    #print (f"First 10 tokens are: {tokens[:10]}")
    #print (f"Decoded back they are: {text[:10]}")
    sequence_length = 16
    d = Dataset(tokens, sequence_length)
    training_set = d.make_ngrams()    
    #print (f"Few training sets are: {training_set[:10]}")

    training_examples = d.get_examples()
    #print (f"Few inputs and target sets are: {training_examples[:10]}")
    print(f"Number of examples: {len(training_examples)}")
    print(f"Each input length:  {len(training_examples[0][0])}")
    print(f"Each target length: {len(training_examples[0][1])}")

    input_shape, target_shape = d.get_shape(training_examples)
    print(f"Input shape is: {input_shape}")
    print(f"Target shape is: {target_shape}")