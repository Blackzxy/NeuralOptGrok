import math
import copy
import sys
import math
from tqdm import tqdm
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import wandb

torch.autograd.set_detect_anomaly(True)

# Amplifier
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
        
class NeuralAmplifier(nn.Module):
    def __init__(self, 
                 input_dim=1, 
                 hidden_dims=[64, 64], 
                 output_dim=1, 
                 dropout_rate=0.1,
                 c_norm=1.0):
        super().__init__()
        layers = []
        prev_dim = input_dim # param-wise: input_dim=1; all_grad: input_dim = concat_grad_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)
        self.c_norm = c_norm
    
    def forward(self, g):
        p = self.network(g) # shape: [g.shape]
        g = self.c_norm * p * g / torch.norm(p * g)
        return g

class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a


        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(nn.Module):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    
    def forward(self, x):
        h = self.token_embeddings(x) # seq_len, batch, dim
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        h = h + self.position_embeddings(positions).expand_as(h)
        
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h) # seq_len, batch, num_tokens
        return logits 
    
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_proj', default='NeuralGrok', type=str)
    parser.add_argument('--wandb_run', default=None, type=str)
    
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dim", type=int,default=128)
    parser.add_argument("--n_layers",type=int, default=2)
    parser.add_argument("--seq_len",type=int, default=5)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--neural_hidden_dims", type=str, default="64,128,64")
    parser.add_argument("--neural_grad", action='store_true')
    parser.add_argument("--tl_eval", action="store_true")
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--track_grad", action="store_true")
    parser.add_argument("--fft", action="store_true")
    parser.add_argument("--device", type=str,default="cuda")

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir", "meta"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=2.0)

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    parser.add_argument("--with_tracking", action="store_true")
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    model = Decoder(
        dim=args.dim, num_layers=args.n_layers, num_heads=args.n_heads, num_tokens=args.p + 4, seq_len=args.seq_len,
    ).to(args.device)
    
    hidden_dims = [int(h) for h in args.neural_hidden_dims.split()]
    amp = NeuralAmplifier(
        input_dim=1,
        hidden_dims=hidden_dims,
        output_dim=1,
    ).to(device)
    
    import pdb
    pdb.set_trace()
    