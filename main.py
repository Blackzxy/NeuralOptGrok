import math
from argparse import ArgumentParser
from itertools import permutations
import copy
import sys
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from grokfast import *
from utils import SVD_Decomp, Nuclear_Norm
import wandb


class NeuralGrad_OneMLP(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=2, alpha=16, beta=6):
        super(NeuralGrad_OneMLP,self).__init__()

        self.alpha = alpha
        self.beta = beta

        hidden_dim_alpha = int(self.alpha * hidden_dim)

        layers = []

        layers.append(nn.Linear(1, hidden_dim_alpha))
        layers.append(nn.ReLU())

        for i in range(n_layers-1):
            if i == n_layers-2:
                layers.append(nn.Linear(hidden_dim_alpha, 1))
            else:
                layers.append(nn.Linear(hidden_dim_alpha, hidden_dim_alpha))
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, grad):
        g1 = self.mlp(grad)
        p = self.softmax(g1)
        x = p * grad
        return x



class NeuralGrad(nn.Module):
    def __init__(self, hidden_dim=32, n_layers=2, alpha=16, beta=6):
        super(NeuralGrad,self).__init__()

        self.alpha = alpha
        self.beta = beta

        hidden_dim_alpha = int(self.alpha * hidden_dim)
        hidden_dim_beta = int(self.beta * hidden_dim)

        layers = []

        layers.append(nn.Linear(1, hidden_dim_alpha))
        layers.append(nn.ReLU())
       

        for i in range(n_layers-1):
            if i == n_layers-2:
                layers.append(nn.Linear(hidden_dim_alpha, 1))
            else:
                layers.append(nn.Linear(hidden_dim_alpha, hidden_dim_alpha))
                layers.append(nn.ReLU())
               
        
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
        # return p * grad

        msk = self.mask2(grad)
        x = p * grad + msk * grad * p
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

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    
    def forward(self, x):
        # print(x.shape)
        # sys.exit(0)
        #print("X: ", x.shape)
        h = self.token_embeddings(x) # seq_len, batch, dim
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        h = h + self.position_embeddings(positions).expand_as(h)


        
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
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


        return torch.stack([a, b, c, op1, op2, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]


class aa_sub_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2)
    
    def generate_data(self, p, eq_token, op_token1, op_token2):
        """
        (a-b+c) % p for 0 <= a, c < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1,p)
        a, b = torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        result = (a * a - b) % p


        return torch.stack([a, b, op1, op2, eq, result])


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
        (a*b) % p for 0 <= a < p, 0< b< p
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

class a_plus_b_minus_ab_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1, op_token2, op_token3):
        self.data = self.generate_data(p, eq_token, op_token1, op_token2, op_token3)
    
    def generate_data(self, p, eq_token, op_token1, op_token2, op_token3):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        op2 = torch.ones_like(a) * op_token2
        op3 = torch.ones_like(a) * op_token3
        result = (a + b - a * b) % p


        return torch.stack([a, b, op1, op2, op3, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

class a_plus_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a + b) % p


        return torch.stack([a, b, op1, eq, result])


    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    
class a_minus_b_mod_p_data(Dataset):

    def __init__(self, p, eq_token, op_token1):
        self.data = self.generate_data(p, eq_token, op_token1)
    
    def generate_data(self, p, eq_token, op_token1):
        """
        (a*b) % p for 0 <= a < p, 0< b< p
        """
        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b= torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq_token
        op1 = torch.ones_like(a) * op_token1
        result = (a - b) % p


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



class MixSimpleDataset(Dataset):
    def __init__(self, p, eq_token, op1, op2, op3):
        self.data = self.generate_data(p, eq_token, op1, op2, op3)
    
    def generate_data(self, p, eq, op1, op2, op3):
        """
        Simple Mix Dataset including:
        1. (a + b) mod p
        2. (a - b) mod p
        3. (a * b) mod p

        op1: +
        op2: -
        op3: *
        """

        a = torch.arange(p)
        b = torch.arange(1, p)

        a, b = torch.cartesian_prod(a, b).T

        eq = torch.ones_like(a) * eq
        op1 = torch.ones_like(a) * op1
        res1 = (a + b) % p

        op2 = torch.ones_like(a) * op2
        res2 = (a - b) % p

        op3 = torch.ones_like(a) * op3
        res3 = (a * b) % p

        d1 = torch.stack([a, b, op1, eq, res1])
        d2 = torch.stack([a, b, op2, eq, res2])
        d3 = torch.stack([a, b, op3, eq, res3])
    
        return torch.cat([d1, d2, d3], dim=1)
    
    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]
    


class MixHardDataset(Dataset):
    def __init__(self,p, eq, op1, op2, op3):
        self.data = self.generate_data(p, eq, op1, op2, op3)
    
    def generate_data(self, p, eq, op1, op2, op3):
        """
        Hard Mix Dataset including:
        1. (a + b - c) mod p
        2. (a * b + c) mod p
        3. (a * b + c * b) mod p
        4. (a - b + c) mod p
        

        op1: +
        op2: -
        op3: *
        """
        a = torch.arange(p)
        b = torch.arange(1, p)
        c = torch.arange(1, p)

        a, b, c = torch.cartesian_prod(a, b, c).T

        eq = torch.ones_like(a) * eq
        op1 = torch.ones_like(a) * op1
        res1 = (a + b - c) % p

        op2 = torch.ones_like(a) * op2
        res2 = (a * b + c) % p

        op3 = torch.ones_like(a) * op3
        res3 = (a * b + c * b) % p
        res4 = (a - b + c) % p

        d1 = torch.stack([a, b, c, op1, op2, eq, res1])
        d2 = torch.stack([a, b, c, op1, op3, eq, res2])
        d3 = torch.stack([a, b, c, op1, op3, eq, res3])
        d4 = torch.stack([a, b, c, op2, op3, eq, res4])

        return torch.cat([d1, d2, d3, d4], dim=1)
    
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

    # for n, p in model.named_parameters():
    #     print(n, p.shape)
    # sys.exit()

    neural_grad = NeuralGrad(
        hidden_dim=args.neural_hidden_dim,
        n_layers=args.neural_layers,
        alpha=args.neural_alpha,
        beta=args.neural_beta,
    ).to(device)

    # ckpt_path = 'results/acc_(ab-cb)mod_p_p=23_AuxLoss_False_TD_2Layers_4Heads_128Dim_lr0.001_NeuralGrad_3NeuralLayers_12Alpha_40Beta_4InnerLoop.pt'
    # neural_grad.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # neural_grad = NeuralGrad_OneMLP(
    #     hidden_dim=args.neural_hidden_dim,
    #     n_layers=args.neural_layers,
    #     alpha=args.neural_alpha,
    #     beta=args.neural_beta,
    # ).to(device)

    if args.tl_eval:
        ### load pretrained ckpt for transfer learning experiments
        ckpt_path = 'results/acc_(ab-cb)mod_p_p=23_AuxLoss_False_TD_2Layers_4Heads_128Dim_lr0.001_NeuralGrad_2NeuralLayers_24Alpha_24Beta_10InnerLoop.pt'
        neural_grad.load_state_dict(torch.load(ckpt_path, weights_only=True))
        neural_grad.eval()
        print("************************LOADED***********************************")

 

    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams/1e6}')
    # sys.exit(0)

    #data = multiplication_mod_p_data(args.p, eq_token, op_token)
    #dataset = ComputationModDataset(args.p, eq_token=eq_token, op_token=op_token, op_token2=op_token2)

    # dataset = ab_sub_cb_mod_p_data(args.p, eq_token, op_token1, op_token2)
    # # dataset_type = '(ab-cb)mod_p'
    # dataset_type = 'Transfer_(a+b)mod_97_to_(ab-cb)mod_p'

    # dataset = aa_sub_b_mod_p_data(args.p, eq_token, op_token1, op_token2)
    # dataset_type = "(aa-b)mod_p"
    # dataset_type = 'ContinueTraining_Transfer_(ab-cb)mod_23_to_(a-bc)mod_p'

    # dataset = ac_plus_bd_sub_e_mod_p_data(args.p, eq_token, op_token1, op_token2, op_token3)
    # # dataset_type = "(ac+bd-e)mod_p"
    # dataset_type = "Transfer_(ab-cb)mod_23_to_(ac+bd-e)mod_p"

    # dataset = MixSimpleDataset(args.p, eq_token, op_token1, op_token2, op_token3)
    # # dataset_type = "MixSimpleData_p"
    # dataset_type = 'Transfer_(ab)mod_97_to_MixSimpleData_p'

    dataset = ab_mod_p_data(args.p, eq_token, op_token1)
    dataset_type = "(ab)mod_p"
    # dataset_type = 'Transfer_(a-b)mod_97_to_(ab)mod_p'
    
    # dataset = a_plus_b_minus_ab_mod_p_data(args.p, eq_token, op_token1, op_token2, op_token3)
    # dataset_type = "(a+b-ab)mod_p"

    # dataset = a_minus_b_mod_p_data(args.p, eq_token, op_token1)
    # dataset_type = "(a-b)mod_p"
    # dataset_type = "Transfer_(a+b)mod_97_to_(a-b)mod_p"

    # dataset = a_plus_b_mod_p_data(args.p, eq_token, op_token1)
    # dataset_type = '(a+b)mod_p'
    # dataset_type = "Transfer_(ab)mod_97_to_(a+b)mod_p"

    #data = expression_mod_p_data(args.p, eq_token, op_token, op_token2)
    
    
    # print(len(dataset), dataset.data.shape, dataset.__getitem__(50))
    # sys.exit(0)

    # train_size = data.shape[1] // 2
    # indices = torch.randperm(data.shape[1])
    # train_idx, valid_idx = indices[:train_size], indices[train_size:]
    # train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    train_size = int(0.5 * len(dataset))
    if args.aux_loss:
        train_size = int(0.45 * len(dataset))
        valid_size = int(0.05 * len(dataset))
    else:
        
        valid_size = int(0 * len(dataset))
    test_size = len(dataset)-train_size-valid_size

    #train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_data, valid_data, test_data = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_data,
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

    if not args.tl_eval:
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
    print(steps_per_epoch, int(args.budget) // steps_per_epoch )

    its, train_acc, test_acc, train_loss, test_loss = [], [], [], [], []

    lastlayer_attn_out_proj_grad_before = []
    lastlayer_attn_out_proj_grad_after = []

    gradients_before = {name: [] for name, param in model.named_parameters() if "layers" in name}
    gradients_after = {name: [] for name, param in model.named_parameters() if "layers" in name}

    grads = None
    i = 0

    # For logging network weights.
    net_its, nets = [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

 
        #for data, is_train in [(train_data, True), (valid_data, False)]:
        for loader, is_train in [(train_loader, True), (test_loader, False)]:

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
                        
                        cur_innerloop_attn_out_proj_grads_before, cur_innerloop_attn_out_proj_grads_after = [], []


                        ## Inner Loop
                        for inner_step_idx in range(inner_steps):

                            with torch.set_grad_enabled(is_train):
                                logits = model(input[:-1])
                                # calculate loss only on the answer part of the equation (last element)
                                loss = F.cross_entropy(logits[-1], input[-1])
                                total_loss += loss.item() * input.shape[-1]
                            
                            model.zero_grad()
                            loss.backward() 

                            for name, param in model.named_parameters():
                                # print(name, param, param.grad)
                                grad = param.grad.view(-1,1)
                                modified_grad = neural_grad(grad) ## Modified the param.grad
                                param.grad = modified_grad.view(param.shape)

                                ## track the gradients and compute the ratio
                                if args.track_grad:
                                    if inner_step_idx == inner_steps - 1 and "layers" in name:
                                        gradients_before[name].append(grad.norm(p=2).item())
                                        gradients_after[name].append(modified_grad.norm(p=2).item())

                                        # if f"layers.{args.n_layers-1}.attn.out_proj.weight" in name:
                                        #     cur_innerloop_attn_out_proj_grads_before.append(
                                        #         grad.norm(p=2).item()
                                        #     )
                                        #     cur_innerloop_attn_out_proj_grads_after.append(
                                        #         modified_grad.norm(p=2).item()
                                        #     )
                                      
                                
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                            optimizer.step()
                            scheduler.step()
                        
                        final_loss = F.cross_entropy(model(input[:-1])[-1], input[-1])
                        #print(final_loss)


                        if args.aux_loss:
                            val_loss = 0
                            for val_input in valid_loader:
                                val_input = val_input.to(device).long().transpose(0,1)
                                val_logits = model(val_input[:-1])
                                val_loss += F.cross_entropy(val_logits[-1], val_input[-1])
                            
                            val_loss /= len(valid_loader)
                    
                            final_loss = torch.abs(final_loss - val_loss) + final_loss
                        
                        print(final_loss)


                        if not args.tl_eval:
                            meta_optimizer.zero_grad()
                            final_loss.backward() ## NOTE: after the backward, the gradients only backward to the model, not the neural_grad

                            ## YOU CAN: check the gradients in neural_grad
                            # for name, param in neural_grad.named_parameters():
                            #     print(name, param, param.grad) --> EXPECTED: None

                            meta_optimizer.step()
                            
                        i += 1

                        # if args.track_grad:
                        #     lastlayer_attn_out_proj_grad_before.append(
                        #         np.mean(cur_innerloop_attn_out_proj_grads_before)
                        #     )
                        #     lastlayer_attn_out_proj_grad_after.append(
                        #         np.mean(cur_innerloop_attn_out_proj_grads_after)
                        #     )
                        
                    else:
                       
                        cur_innerloop_attn_out_proj_grads_before = []

                        with torch.set_grad_enabled(is_train):
                            logits = model(input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            loss = F.cross_entropy(logits[-1], input[-1])
                            total_loss += loss.item() * input.shape[-1]
                        
                        model.zero_grad()
                        loss.backward()
                        
                        if args.track_grad:
                            for name, param in model.named_parameters():
                                    # print(name, param, param.grad)
                                    grad = param.grad.view(-1,1)
                                    ## track the gradients and compute the ratio
                                    if  "layers" in name:
                                        gradients_before[name].append(grad.norm(p=2).item())
                                        
                                        # if f"layers.{args.n_layers-1}.attn.out_proj.weight" in name:
                                        #     cur_innerloop_attn_out_proj_grads_before.append(
                                        #         grad.norm(p=2).item()
                                        #     )
                                

                        

                        #######

                        trigger = i < 500 if args.two_stage else False

                        if args.filter == "none":
                            pass
                        elif args.filter == "ma":
                            grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                            
                            if args.track_grad:
                                for name, param in model.named_parameters():
                                    # print(name, param, param.grad)
                                    grad = param.grad.view(-1,1)
                                    ## track the gradients and compute the ratio
                                    if "layers" in name:
                                        gradients_after[name].append(grad.norm(p=2).item())

                        elif args.filter == "ema":
                            grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                        else:
                            raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                        #######

                        optimizer.step()
                        scheduler.step()
            
                        i += 1

                        # if args.track_grad:
                        #     lastlayer_attn_out_proj_grad_before.append(
                        #         np.mean(cur_innerloop_attn_out_proj_grads_before)
                        #     )
                    


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
                test_acc.append(total_acc / len(test_loader.dataset))
                test_loss.append(total_loss / len(test_loader.dataset))
                if args.with_tracking:
                    wandb.log(
                        {
                            "test_acc": total_acc / len(test_loader.dataset),
                            "test_loss": total_loss / len(test_loader.dataset),
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
            do_save = (e + 1) % 10 == 0
           
        if do_save:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            print(steps[0], steps[-1])
            print(len(steps), len(train_acc))

            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, test_acc, label="test")

            

            plt.legend()
            plt.title(f"{dataset_type}_p={args.p} (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()
            if args.neural_grad:
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.inner_steps}InnerLoop.png", dpi=150)
            
                if not args.tl_eval:
                    torch.save(neural_grad.state_dict(), f"results/acc_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.inner_steps}InnerLoop.pt")
            elif args.filter:
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.png", dpi=150)
                #torch.save(neural_grad.state_dict(), f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.pt")
            else:
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}.png", dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, test_loss, label="test")
            plt.legend()
            plt.title(f"{dataset_type}_p={args.p}_(training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid()
            if args.neural_grad:
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.inner_steps}InnerLoop.png", dpi=150)
            elif args.filter:
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.png", dpi=150)
            else:
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}.png", dpi=150)
          
            plt.close()

            if args.track_grad:
                n_params = len(gradients_before)
                cols = 4
                rows = math.ceil(n_params / cols)

                if args.neural_grad:
                    fig, axes = plt.subplots(rows, cols, figsize=(24, 6*rows))
                    axes = axes.flatten()

                    for idx, (name, grads_list) in enumerate(gradients_after.items()):
                        axes[idx].plot(grads_list)
                        axes[idx].set_title(f"{name}", fontsize=15)
                        axes[idx].set_xlabel("Optimization Steps", fontsize=15)
                        axes[idx].set_ylabel("Gradient Norm")
                        axes[idx].set_xscale('log', base=10)
                        axes[idx].grid(True)
                    
                    for ax in axes[n_params:]:
                        ax.axis("off")
                    
                    plt.savefig(f"results/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.inner_steps}InnerLoop.png", dpi=250)
                    plt.close()

                else:
                    fig, axes = plt.subplots(rows, cols, figsize=(24, 6*rows))
                    axes = axes.flatten()
                    if args.filter:
                        grads_list = gradients_after
                    else:
                        grads_list = gradients_before

                    for idx, (name, grads_list) in enumerate(grads_list.items()):
                        axes[idx].plot(grads_list)
                        axes[idx].set_title(f"{name}", fontsize=15)
                        axes[idx].set_xlabel("Optimization Steps", fontsize=15)
                        axes[idx].set_ylabel("Gradient Norm")
                        axes[idx].set_xscale('log', base=10)
                        axes[idx].grid(True)
                    

                    for ax in axes[n_params:]:
                        ax.axis("off")
                    
                    plt.savefig(f"results/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.png", dpi=250)
                    plt.close()

                # plt.plot(lastlayer_attn_out_proj_grad_before, label="out_proj_grad_before")
                # if args.neural_grad:
                #     plt.plot(lastlayer_attn_out_proj_grad_after, label="out_proj_grad_after")
                # plt.legend()
                # plt.title(f"{dataset_type}_p={args.p}_(training on 50% of data)")
                # plt.xlabel("Optimization Steps")
                # plt.ylabel("Gradient Norm")
                # plt.xscale("log", base=10)
                # plt.grid()
                # if args.neural_grad:
                #     plt.savefig(f"results/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.inner_steps}InnerLoop.png", dpi=150)
                # else:
                #     plt.savefig(f"results/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.png", dpi=150)
                # plt.close()


            # results = {
            #     'its': its,
            #     'train_acc': train_acc,
            #     'train_loss': train_loss,
            #     'val_acc': val_acc,
            #     'val_loss': val_loss,
            # }

            # if args.save_weights:
            #     net_its.append(e)
            #     nets.append(copy.deepcopy(model.state_dict()))
            #     results['net_its'] = net_its
            #     results['net'] = nets

            # torch.save(results, f"results/res_{args.label}.pt")


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
    parser.add_argument("--tl_eval", action="store_true")
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--track_grad", action="store_true")

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
