from reader import read_file, read_wikitext
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text)) #removes duplicates then sorts so ids will be assigned to a sorted list
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        # converts each character to its Unicode code point
        #return [ord(c) for c in text] #this puts ascii value of each character which will make varying look up tables
        return [self.char_to_id[c] for c in text]

    def decode(self, tokens):
        return ''.join(self.id_to_char[t] for t in tokens)
        #return [self.id_to_char[c] for c in tokens]
    
    def vocab_size(self, text):
        return len(set(text))
    
if __name__ == "__main__":
    content = read_wikitext()
    t = Tokenizer(content)
    tokens = t.encode(content)
    text = t.decode(tokens)
    size = t.vocab_size(text)
    print ("Vocab size is", size)
    print (f"Total tokens in dataset: {len(tokens):,}")
    print (f"First 10 tokens are: {tokens[:10]}")
    print (f"Decoded back they are: {text[:10]}")
