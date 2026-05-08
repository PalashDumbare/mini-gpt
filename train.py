import torch
import torch.nn as nn
from dataset  import *
from model import GPT
from config import *

text = load_data()
stoi,itos = build_vocab(text)
vocab_size = len(stoi)

data = torch.tensor(encode(text,stoi),dtype=torch.long)

model = GPT(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
loss_fn = nn.CrossEntropyLoss()

for iter in range(MAX_ITERS):
    print(iter)
    x,y = get_batch(data, batch_size=BATCH_SIZE)
    x,y = x.to(DEVICE),y.to(DEVICE)
    
    logits = model(x)

    # logits → (B, T, V)
    # y     → (B, T)
    # Not compatible with CrossEntropyLoss
    # Fix → Flattening
    # logits	(B*T, V)
    # target	(B*T)
    loss = loss_fn(
        logits.view(-1,vocab_size), # -> (B*T, V)
        y.view(-1)                  # -> (B*T)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {iter}, Loss: {loss.item()}")
    
torch.save(model.state_dict(), "model.pt")

