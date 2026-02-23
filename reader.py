#!pip install datasets
from datasets import load_dataset

def read_file(path):
    with open(path, 'r') as f:
        return f.read()
    
def read_wikitext():
    #dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = dataset['train']['text'] #loads text from training data
    full_text = '\n'.join(text)
    return full_text

if __name__ == "__main__":
    content = read_wikitext()
    #print(content)
