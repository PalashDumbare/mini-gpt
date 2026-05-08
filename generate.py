import torch
from model import GPT
from dataset import *
from config import *

text = load_data()
stoi, itos = build_vocab(text)

model = GPT(len(stoi))
model.load_state_dict(torch.load("model.pt",map_location = DEVICE))
model.to(DEVICE)
model.eval()

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = float('-inf')
    return logits

def generate(start, max_tokens = 100):
    x = torch.tensor(encode(start,stoi)).unsqueeze(0).to(DEVICE)
    temperature = 0.8
    k = 2
    for _ in range(max_tokens):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        logits = top_k_logits(logits, k)
        probs = torch.softmax(logits ,dim = -1)
        next_token = torch.multinomial(probs,1)
        x = torch.cat([x, next_token], dim=1)
    return decode(x[0].tolist(), itos)

print(generate("What is the canst the "))