import requests
import torch
from config import BATCH_SIZE

def load_data():
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    response = requests.get(url).text
    print(f"Data loaded from {url}, length: {len(response)} characters")
    build_vocab(response)
    return response

def build_vocab(text):
    text_set = set(text)
    chars = sorted(list(text_set))
    stoi = {ch : i  for i, ch in enumerate(chars)}
    itos = {i : ch for i,ch in enumerate(chars)}
    return stoi,itos

def encode(text,stoi):
    return [stoi[ch] for ch in text]

def decode(text,itos):
    return ''.join([itos[i] for i in text])

def get_batch(data,batch_size):
    # Generate random sampling between the range of (0 to len(data)-BLOCK_SIZE)
    # (batch_size,) : This will generate 1D tensor with  batch_size elements
    print(len(data))
    random_sample = torch.randint(0, len(data)-BATCH_SIZE, (batch_size,))
    # [data[i:i+BLOCK_SIZE] : Creates a list of tensors
    # stack concatenates them to a single tensor 
    # After Stack -> new shape ((batch_size, BLOCK_SIZE))
    # Neural networks (like GPT) expect input in batch form: (batch_size, sequence_length)
    x = torch.stack([data[i:i+BATCH_SIZE] for i in random_sample])
    y = torch.stack([data[i+1:i+BATCH_SIZE+1] for i in random_sample]) # Used for target labels, shifted by 1
    return x,y




