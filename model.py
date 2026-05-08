import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(EMB_DIM,EMB_DIM)
        self.W_key = nn.Linear(EMB_DIM,EMB_DIM)
        self.W_value = nn.Linear(EMB_DIM,EMB_DIM)
        self.fc =  nn.Linear(EMB_DIM,EMB_DIM)

        # triu : Keeps upper triangle ABOVE diagonal
        self.register_buffer("mask",torch.triu(torch.ones(BATCH_SIZE,BATCH_SIZE), diagonal = 1))


    def forward(self,x):
        batch_sze, token_len, dim = x.shape
        keys = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        # Match Making
        # Shape of k,q,v = (batch_sze,token_len,dim) 
        # Transpose of (B,T,D) ->  (B,D,T)
        # After Transpose multiply : (B,T,D) @ (B,D,T) = (B,T,T)   
        # (T,T) : Each token compared with every other token
        attn_score = query @ keys.transpose(1,2)

        # casual masking
        # ensures model cannot "see future tokens" (autoregressive property)
        # As per register_buffer above, self.mask is a fixed sqaure  matrix of (CONTEXT_LEN*CONTEXT_LEN)
        # slicing is needed as for some input we might not use full content_length
        # ex : Context_Length = 10, and senetence is just 3 token, so we need 3*3  not 10*10
        # Takes top left portion. Original (10×10) → Slice → (4×4)

#        mask_bool =
#       [[False, True,  True,  True ],
#       [False, False, True,  True ],
#       [False, False, False, True ],
#       [False, False, False, False]]
        mask_bool = self.mask[:token_len,:token_len].bool()
        
        # Allowed positions → unchanged
        # Blocked positions → -∞
        attn_scores = attn_score.masked_fill(mask_bool,-torch.inf)

        # attn_scores.shape = (batch_size, T, T) : For single head 
        # (T,T) : Each token compared with every other token
        # dim = -1 : which tokens to attend to
        # dim = -1 → normalize across tokens (attention distribution per token)
        # Each row becomes a probability distribution (sums to 1)
        attn_weights = torch.softmax(attn_scores/(EMB_DIM**0.5),dim=-1)

        context_vec = attn_weights @ value

        return context_vec
    
class TransformerBloc(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = nn.Sequential(
            nn.Linear(EMB_DIM,4*EMB_DIM),
            nn.ReLU(),
            nn.Linear(4*EMB_DIM,EMB_DIM)
        )
        self.layer_norm1 = nn.LayerNorm(EMB_DIM)
        self.layer_norm2 = nn.LayerNorm(EMB_DIM)
    
    def forward(self,x):
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x


class GPT(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
#       input -> token ids, output = each token converted into a vector
#       creates a trainable lookup table (matrix)
#       Each token ID (word/subword) → mapped to a vector
#       Token 0 → [0.1, 0.3, -0.2, 0.5...] 
#       Token 1 → [0.7, -0.1, 0.8, 0.2...]
#       Token 2 → [0.4, 0.9, -0.6, 0.1...]
#         ...
#       Token 69 → [0.4, 0.9, -0.6, 0.1...]
#       Ex : "hello" → 42 → [0.12, -0.8, 0.55, ...] :  
#       NN can't work with number, it needs continous vectors, so we convert number to vector
        self.tok_emb = nn.Embedding(vocab_size,EMB_DIM)

#       CONTEXT_LEN → “how far model can remember”
#       this layer creates a look up table for positions ex for (6,4)
#       Position 0 → [0.12, -0.4, 0.8, 0.3]
#       Position 1 → [0.91,  0.2, -0.1, 0.5]
#       Position 2 → [0.33,  0.7,  0.6, 0.9]
#         ...
#       Position 5 → [...]
        self.pos_emb = nn.Embedding(BATCH_SIZE,EMB_DIM)

        self.blocks = nn.Sequential(
            *[TransformerBloc() for _ in range(NUM_LAYERS)] # *[Block(), Block(), Block()] unpacks to Block(), Block(), Block() 
        )
        self.ln_f = nn.LayerNorm(EMB_DIM)
        self.output_layer = nn.Linear(EMB_DIM, vocab_size)

    def forward(self,in_indx):
        # in_indx is the list of tokens of each sentence ex : [10, 45, 22, 18, 56, 31, 14]
        # batch_size,context_len
        batch_size, token_len  = in_indx.shape

        # Now computers cannot calculate using no 10, they need vectors
        # Token Embedding layer looks up each ID in a giant table, and lets say 10(She) is converted 
        # to vector "She" -> [0.1, -0.5, 0.2, ... 24]
        # So we can say, this layer converts each token into vector of n dim.
        tok_embeds = self.tok_emb(in_indx)

        # creates a list of numbers: [0, 1, 2, 3, 4, 5, 6, 7] -> len of sentence
        # pos embed layer create a separate vector for each slot (POS0,POS2...)
        # Position 0 (where "Word1" is) -> [0.01, 0.04, ...]
        # Position 1 (where "Word2" is) -> [0.88, -0.1, ...]
        pos_embeds = self.pos_emb(torch.arange(token_len).to(in_indx.device)) 
        
        # Final Vector =  Meaning (Token)  +  Location (Position) 
        # Now, the vector for "Word1" contains both the fact that it is a pronoun/xyz and that it is the first word in the sentence.
        x = tok_embeds + pos_embeds
        x = self.blocks(x)
        x = self.ln_f(x)

        # (4, 6, 70)-> 4 sentences, 6 words each, 70 possible word scores for every word
        # produces Logits (scores) for every possible word it knows.
        # takes that combined vector and maps it back to the size of your vocabulary
        logits = self.output_layer(x) # Logits Shape: (Batch, token_length, Vocab_Size)
        
        return logits


        


















        