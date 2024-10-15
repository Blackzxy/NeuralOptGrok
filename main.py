import math
from argparse import ArgumentParser
from itertools import permutations
import copy
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from grokfast import *
import wandb

class NeuralGrad(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=2, alpha=16, beta=6):
        super(NeuralGrad,self).__init__()

        self.alpha = alpha
        self.beta = beta

        hidden_dim_alpha = int(self.alpha * hidden_dim)
        hidden_dim_beta = int(self.beta * hidden_dim)

        layers = []

        layers.append(nn.Linear(1, hidden_dim_alpha))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.1))

        for i in range(n_layers-1):
            if i == n_layers-2:
                layers.append(nn.Linear(hidden_dim_alpha, 1))
            else:
                layers.append(nn.Linear(hidden_dim_alpha, hidden_dim_alpha))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers)


        layers = []

        layers.append(nn.Linear(1, hidden_dim_beta))
        layers.append(nn.ReLU())

        for i in range(n_layers-1):
            if i == n_layers-2:
                layers.append(nn.Linear(hidden_dim_beta, 1))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_dim_beta, hidden_dim_beta))
                layers.append(nn.ReLU())

        self.mask2 = nn.Sequential(*layers)
        
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, grad):
        g1 = self.mlp(grad)
        p = self.softmax(g1)
        msk = self.mask2(grad)
        x = p * grad + msk * grad * p
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.GELU(),
        nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        """
        x: (seq_len, batch_size, dim)
        """
        # Self-attention (preserves shape)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output) # Residual connection
        # Feedforward network (preserves shape)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Residual connection
        return x # Shape: (seq_len, batch_size, dim)

class MemoryModule(nn.Module):
    def __init__(self, memory_size, dim, num_heads=4):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        # Memory initialized as learnable parameters
        self.memory = nn.Parameter(torch.zeros(memory_size, dim), requires_grad=True)

    def read(self, query):
        ## query: 1, dim x memory.T: dim, memory_size -> 1, ms
        attn_weights = F.softmax(torch.matmul(query, self.memory.T), dim=-1)
        return torch.matmul(attn_weights, self.memory) # 1, dim

    def write(self, input_vector):
        self.memory.data += input_vector.mean(dim=0).data.unsqueeze(0)


class StatefulTransformerDecoder(nn.Module):
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=100, seq_len=10, memory_size=5):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(num_tokens, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)

        # Transformer Decoder blocks
        self.layers = nn.ModuleList([TransformerDecoderBlock(dim, num_heads) for _ in range(num_layers)])

        # Memory module
        self.memory_module = MemoryModule(memory_size, dim, num_heads)

        # Output projection
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, num_tokens)

    def forward(self, x):
        """
        x: (seq_len, batch_size)
        """
        seq_len = x.size(0)
        batch_size = x.size(1)

        # Get token embeddings
        x = self.token_embedding(x) # Shape: (seq_len, batch_size, dim)

        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(1) # Shape: (seq_len, 1)
        x = x + self.position_embedding(position_ids).expand_as(x) # Shape: (seq_len, batch_size, dim)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x) # Shape: (seq_len, batch_size, dim)

            # Memory interaction after each layer
            memory_output = self.memory_module(x)
            x = x + memory_output # Shape: (seq_len, batch_size, dim)

        # Update memory using the mean of the current state
        current_state = x.mean(dim=0) # Shape: (batch_size, dim)
        self.memory_module.update_memory(current_state)

        # Final normalization and projection
        x = self.norm(x) # Shape: (seq_len, batch_size, dim)
        logits = self.fc_out(x) # Shape: (seq_len, batch_size, num_tokens)

        return logits


class RecurrentMemory(nn.Module):
    def __init__(self, memory_size, dim):
        super(RecurrentMemory, self).__init__()
        self.memory_size = memory_size
        self.dim = dim

        self.memory = nn.Parameter(torch.zeros(memory_size, dim), requires_grad=False)

    
    def forward(self, hidden_states):
        '''
        hidden_states: seq_len, batch_size, dim
        '''
        new_memory = torch.cat([self.memory, hidden_states], dim=0)
        new_memory = new_memory[-self.memory_size:]
        self.memory = nn.Parameter(new_memory.detach(), requires_grad=False)
        return self.memory

class RecurrentMemoryTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super(RecurrentMemoryTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
       
    
    def forward(self, x, memory):
        '''
        x: seq_len, batch_size, dim
        memory: memory_size, batch_size, dim
        '''
        print(memory.shape, x.shape)
        memory = memory.unsqueeze(1)
        combined_input = torch.cat([memory, x], dim=0) # seq_len+memory_size, batch_size, dim

        attn_output, _ = self.self_attn(combined_input, combined_input, combined_input)

        attn_output = attn_output[-x.size(0):] # seq_len, batch_size, dim

        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.ff(x)

        x = x + ff_output
        x = self.norm2(x)

        return x

class RecurrentMemoryTransformer(nn.Module):
    def __init__(self, num_tokens, dim, num_heads, ff_dim, num_layers, memory_size, seq_len):
        super(RecurrentMemoryTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.transformer_layer = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.memory_module = MemoryModule(memory_size,dim)
        self.fc_out = nn.Linear(dim, num_tokens)
    

    def forward(self, x):
        x = self.embedding(x) # seq_len, batch_size, dim

        memory_state = torch.zeros(1, x.size(-1)).to(x.device) # 1, dim

        for t in range(x.size(0)):
            memory_read = self.memory_module.read(memory_state) # 1, dim

            combined_input = x[t] + memory_read # batch, dim

            for layer in self.transformer_layer:
                combined_input = layer(combined_input.unsqueeze(0)).squeeze(0) # batch dim
            
            memory_state = combined_input
            self.memory_module.write(combined_input)
        
        output = self.fc_out(combined_input)
        return output


class RecurrentTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super(RecurrentTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x, memory):
        combined_input = torch.cat([memory, x], dim=0)

        attn_output, _ = self.self_attn(combined_input, combined_input, combined_input)

        attn_output = attn_output[-x.size(0):]

        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.ff(x)

        x = x + ff_output
        x = self.norm2(x)

        return x




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

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5, memory_size=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))

        # self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

        # self.memory_module = MemoryModule(memory_size=memory_size,dim=dim)
        # self.memory_state = nn.Parameter(torch.randn(1, dim), requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        # sys.exit(0)
        #print("X: ", x.shape)
        h = self.token_embeddings(x) # seq_len, batch, dim
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        h = h + self.position_embeddings(positions).expand_as(h)

        #print("H: ", h.shape)

        
        #print("Memory: ", memory_state.shape)

        # for t in range(h.size(0)): # seq_len
        #     memory_read = self.memory_module.read(self.memory_state) # 1, dim
            

        #     combined_input = h[t] + memory_read # batch, dim

        #     for layer in self.layers:
        #         combined_input = layer(combined_input.unsqueeze(0)).squeeze(0) # batch dim
            
            
        #     self.memory_state = self.memory_state = nn.Parameter(self.memory_state.data + 0.5 * combined_input.mean(dim=0, keepdim=True).detach())
        #     self.memory_module.write(combined_input)

 

        # h = self.ln_f(combined_input) #batch, dim
        # #print("H: ", h.shape)
        # logits = self.head(h)

        for layer in self.layers:
            h = layer(h)

        logits = self.head(h) # seq_len, batch, num_tokens
        # print("Logit: ", logits.shape)
        # sys.exit(0)
        return logits 


def multiplication_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])

def expression_mod_p_data(p, eq_token, op_token1, op_token2):
    """(a*b + c*d) % p for 0 ≤ a, b, c, d < p
    """
    a = torch.arange(p)
    b = torch.arange(p)
    c = torch.arange(p)
    d = torch.arange(p)
    a, b, c, d = torch.cartesian_prod(a, b, c, d).T

    eq = torch.ones_like(a) * eq_token
    op1 = torch.ones_like(a) * op_token1
    op2 = torch.ones_like(a) * op_token2
    result = (a * b + c * d) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b + c◦d = e, where each of “a”, “◦”, “b”, “+”, “c”, “◦”, “d”, “=”, and “e”
    # is a separate token"
    return torch.stack([a, op1, b, op2, c, op1, d, eq, result])


class ab_sub_cb_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2)
    
    def generate_data(self, p, eq_token, op_token1, op_token2):
        """
        (a*b-c*b) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1,p)
        c = torch.arange(p)
        a, b, c = torch.cartesian_prod(a, b, c).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        result = (a * b - c * b) % p


        return torch.stack([a, op1, b, op2, c, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
class ac_plus_bd_sub_e_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2, op_token3):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2, op_token3)
    
    def generate_data(self, p, eq_token, op_token1, op_token2, op_token3):
        """
        (a*c+b*d-e) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(p)
        c = torch.arange(1, p)
        d = torch.arange(1, p)
        e = torch.arange(p)
        a, b, c, d, e = torch.cartesian_prod(a, b, c, d, e).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        op3 = torch.ones_like(a) * op_token3
        result = (a * c + b * d - e) % p


        return torch.stack([a, b, c, d, e, op1, op2, op3, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

class ab_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*c+b*d-e) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a * b) % p


        return torch.stack([a, b, op1, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

class ComputationModDataset(Dataset):

    def __init__(self, p, eq_token, op_token, op_token2):
        self.data = self.generate_data(p, eq_token, op_token, op_token2)
    
    def generate_data(self, p, eq_token, op_token, op_token2):
        # x = torch.arange(p)
        # y = torch.arange(1,p)
        # x,y = torch.cartesian_prod(x,y).T

        # eq = torch.ones_like(x) * eq_token
        # op = torch.ones_like(x) * op_token
        # result = x * y % p

        # return torch.stack([x, op, y, eq, result])

        a = torch.arange(p)
        b = torch.arange(1, p)
        c = torch.arange(1, p)
        a, b, c= torch.cartesian_prod(a, b, c).T
        # d = torch.arange(1, p)
        # a, b, c, d = torch.cartesian_prod(a, b, c, d).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token
        op2 = torch.ones_like(a) * op_token2
        result = (a * b  + c * b - a*c ) % p

        # "All of our experiments used a small transformer trained on datasets of
        # equations of the form a◦b + c◦d = e, where each of “a”, “◦”, “b”, “+”, “c”, “◦”, “d”, “=”, and “e”
        # is a separate token"
        return torch.stack([a, op1, b, op2, c, eq, result])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token1 = args.p + 1
    op_token2 = args.p + 2
    op_token3 = args.p + 3

    inner_steps = args.inner_steps



    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"

    model = Decoder(
        dim=args.dim, num_layers=args.n_layers, num_heads=args.n_heads, num_tokens=args.p + 2, seq_len=args.seq_len,
        memory_size=args.memory_size
    ).to(device)

    neural_grad = NeuralGrad(
        hidden_dim=args.neural_hidden_dim,
        n_layers=args.neural_layers,
        alpha=args.neural_alpha,
        beta=args.neural_beta,
    ).to(device)

    # model = StatefulTransformerDecoder(
    #     dim=args.dim, num_layers=args.n_layers, num_heads=args.n_heads,
    #     num_tokens=args.p + 2,
    #     seq_len= args.seq_len,
    #     memory_size=16
    # ).to(device)

    # model = RecurrentMemoryTransformer(
    #     num_tokens=args.p+2,
    #     dim=args.dim,
    #     num_heads=args.n_heads,
    #     ff_dim=int(args.dim*4),
    #     num_layers=args.n_layers,
    #     memory_size=args.memory_size,
    #     seq_len=args.seq_len
    # ).to(device)

    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams/1e6}')
    # sys.exit(0)

    #data = multiplication_mod_p_data(args.p, eq_token, op_token)
    #dataset = ComputationModDataset(args.p, eq_token=eq_token, op_token=op_token, op_token2=op_token2)
    #dataset = ab_sub_cb_mod_p_data(args.p, eq_token, op_token1, op_token2)
    #dataset = ac_plus_bd_sub_e_mod_p_data(args.p, eq_token, op_token1, op_token2, op_token3)
    dataset = ab_mod_p_data(args.p, eq_token, op_token1)
    #data = expression_mod_p_data(args.p, eq_token, op_token, op_token2)
    print(len(dataset))

    # train_size = data.shape[1] // 2
    # indices = torch.randperm(data.shape[1])
    # train_idx, valid_idx = indices[:train_size], indices[train_size:]
    # train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    train_size = int(0.5 * len(dataset))
    valid_size = len(dataset)-train_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False)


    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        [
            {
                "params": model.parameters(),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas":(args.beta1, args.beta2)
        },
    
        ]
    )

    meta_optimizer = getattr(torch.optim, args.optimizer)(
        [
            {
                "params": neural_grad.parameters(),
        "lr": 1e-4,
        "weight_decay": args.weight_decay,
        "betas":(args.beta1, args.beta2)
        },
    
        ]
    )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    #steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)
    steps_per_epoch = len(train_loader)

    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    grads = None
    i = 0

    # For logging network weights.
    net_its, nets = [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

 
        #for data, is_train in [(train_data, True), (valid_data, False)]:
        for loader, is_train in [(train_loader, True), (valid_loader, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            # dl = torch.split(data, args.batch_size, dim=1)
            # for input in dl:
            #     input = input.to(device)
            for input in loader:
                input = input.to(device).long().transpose(0,1)
                  # Debugging prints
                # print(f"Input shape: {input.shape}")
                # print(f"Input values: {input}")
                # print(f"Max index: {input.max().item()}, Min index: {input.min().item()}")
                # sys.exit(0)
                if is_train:

                    if args.neural_grad:
                        for _ in range(inner_steps):

                            with torch.set_grad_enabled(is_train):
                                logits = model(input[:-1])
                                # calculate loss only on the answer part of the equation (last element
                                loss = F.cross_entropy(logits[-1], input[-1])
                                total_loss += loss.item() * input.shape[-1]
                            
                            model.zero_grad()
                            loss.backward() #(retain_graph=True)

                            for name, param in model.named_parameters():
                                # print(name, param, param.grad)
                                grad = param.grad.view(-1,1)
                                modified_grad = neural_grad(grad)
                                param.grad = modified_grad.view(param.shape)
                            
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                            # #######

                            # trigger = i < 500 if args.two_stage else False

                            # if args.filter == "none":
                            #     pass
                            # elif args.filter == "ma":
                            #     grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                            # elif args.filter == "ema":
                            #     grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                            # else:
                            #     raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                            # #######

                            optimizer.step()
                            scheduler.step()
                        
                        final_loss = F.cross_entropy(model(input[:-1])[-1], input[-1])
                        print(final_loss)
                        meta_optimizer.zero_grad()
                        final_loss.backward()
                        meta_optimizer.step()
                        
                        i += 1
                    
                    else:
                       

                        with torch.set_grad_enabled(is_train):
                            logits = model(input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            loss = F.cross_entropy(logits[-1], input[-1])
                            total_loss += loss.item() * input.shape[-1]
                        
                        model.zero_grad()
                        loss.backward()

                        

                        #######

                        trigger = i < 500 if args.two_stage else False

                        if args.filter == "none":
                            pass
                        elif args.filter == "ma":
                            grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                        elif args.filter == "ema":
                            grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                        else:
                            raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                        #######

                        optimizer.step()
                        scheduler.step()
            
                        i += 1
                    


                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]


                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]
            
            if is_train:
                train_acc.append(total_acc / len(train_loader.dataset))
                train_loss.append(total_loss / len(train_loader.dataset))
                its.append(i)
                if args.with_tracking:
                    wandb.log(
                    {
                        "train_acc": total_acc / len(train_loader.dataset),
                        "train_loss": total_loss / len(train_loader.dataset),
                    }
                )
            else:
                val_acc.append(total_acc / len(valid_loader.dataset))
                val_loss.append(total_loss / len(valid_loader.dataset))
                if args.with_tracking:
                    wandb.log(
                        {
                            "val_acc": total_acc / len(valid_loader.dataset),
                            "val_loss": total_loss / len(valid_loader.dataset),
                        }
                    )
            # if is_train:
            #     train_acc.append(total_acc / train_data.shape[-1])
            #     train_loss.append(total_loss / train_data.shape[-1])
            #     its.append(i)
            #     wandb.log(
            #         {
            #             "train_acc":total_acc / train_data.shape[-1],
            #             "train_loss":total_loss / train_data.shape[-1],
            #         }
            #     )
            # else:
            #     val_acc.append(total_acc / valid_data.shape[-1])
            #     val_loss.append(total_loss / valid_data.shape[-1])
            #     wandb.log(
            #         {
            #             "val_acc":total_acc / valid_data.shape[-1],
            #             "val_loss":total_loss / valid_data.shape[-1],
            #         }
            #     )

        if args.save_weights:
            do_save = e <= 500 or (e > 500 and (e + 1) % 100 == 0) or e == int(args.budget) // steps_per_epoch - 1
        else:
            do_save = (e + 1) % 100 == 0
        if do_save:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.legend()
            plt.title("Modular Multiplication (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/acc_{args.label}.png", dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title("Modular Multiplication (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/loss_{args.label}.png", dpi=150)
            plt.close()

            results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }

            if args.save_weights:
                net_its.append(e)
                nets.append(copy.deepcopy(model.state_dict()))
                results['net_its'] = net_its
                results['net'] = nets

            torch.save(results, f"results/res_{args.label}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
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
    parser.add_argument("--memory_size", type=int, default=32)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--neural_hidden_dim", type=int, default=32)
    parser.add_argument("--neural_layers", type=int, default=3)
    parser.add_argument("--neural_alpha", type=int, default=16)
    parser.add_argument("--neural_beta",type=int, default=6)
    parser.add_argument("--neural_grad", action='store_true')

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir", "meta"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    parser.add_argument("--with_tracking", action="store_true")
    args = parser.parse_args()

    if args.with_tracking:
        wandb.init(
            project = '(a*b+c*d)modp'
        )

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == "meta":
        filter_suffix = "meta"
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
