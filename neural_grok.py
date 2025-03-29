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


torch.autograd.set_detect_anomaly(True)




class NeuralGrad(nn.Module):
    def __init__(self, hidden_dim=32, n_layers=2, alpha=16):
        super(NeuralGrad,self).__init__()

        self.alpha = alpha
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
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()


    
    def forward(self, grad):
        
        
        mlp1 = self.mlp(grad)
        
        #g1 = mlp1 * grad / torch.norm(grad)  # / torch.norm(mlp1 * grad)
        p = self.softmax(mlp1)
       
        g1 = p * grad / torch.norm(p * grad)
        
        
        x =  g1 
        
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

    t = args.t



    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"

    model = Decoder(
        dim=args.dim, num_layers=args.n_layers, num_heads=args.n_heads, num_tokens=args.p + 4, seq_len=args.seq_len,
        memory_size=args.memory_size
    ).to(device)

    # for n, p in model.named_parameters():
    #     print(n, p.shape)
    # sys.exit()

    neural_grad = NeuralGrad(
        hidden_dim=args.neural_hidden_dim,
        n_layers=args.neural_layers,
        alpha=args.neural_alpha,
    ).to(device)

   

    if args.tl_eval:
        ### load pretrained ckpt
        ckpt_path = 'results/acc_(a+b)mod_p_p=97_AuxLoss_True_TD_2Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_3NeuralLayers_4Alpha_3InnerLoop.pt'
        #"results/acc_(ac+bd-e)mod_p_p=7_AuxLoss_True_TD_4Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_2NeuralLayers_36Alpha_2InnerLoop.pt"
        #'results/acc_(aa-b)mod_p_p=97_AuxLoss_True_TD_2Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_2NeuralLayers_8Alpha_4InnerLoop.pt'
        #'results/acc_(ab)mod_p_p=97_AuxLoss_True_TD_2Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_2NeuralLayers_10Alpha_6Beta_3InnerLoop.pt'
        #'results/acc_(a-b)mod_p_p=97_AuxLoss_True_TD_2Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_2NeuralLayers_18Alpha_3InnerLoop.pt'
        #'results/acc_(a+b)mod_p_p=97_AuxLoss_True_TD_2Layers_4Heads_128Dim_lr0.001_wd0.001_NeuralGrad_3NeuralLayers_4Alpha_3InnerLoop.pt'
        neural_grad.load_state_dict(torch.load(ckpt_path, weights_only=True))
        neural_grad.eval()
        print("************************LOADED***********************************")

    

    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams/1e6}')
    # sys.exit(0)



    # dataset = aa_sub_b_mod_p_data(args.p, eq_token, op_token1, op_token2)
    # dataset_type = "(aa-b)_mod_97"
    # dataset_type = 'Transfer_(ac+bd-e)mod7_to_(aa-b)mod_p'

    dataset = ac_plus_bd_sub_e_mod_p_data(args.p, eq_token, op_token1, op_token2, op_token3)
    dataset_type = "(ac+bd-e)_mod_7"
    # dataset_type = "Transfer_(a-b)mod97_to_(ac+bd-e)mod_p"

    # dataset = ab_mod_p_data(args.p, eq_token, op_token1)
    # dataset_type = "(ab)_mod_97"
    # dataset_type = 'Transfer_(ac+bd-e)mod7_to_(ab)mod_p'
    

    # dataset = a_minus_b_mod_p_data(args.p, eq_token, op_token1)
    # dataset_type = "(a-b)_mod_97"
    # dataset_type = "Transfer_(ac+bd-e)mod97_to_(a-b)mod_p"

    # dataset = a_plus_b_mod_p_data(args.p, eq_token, op_token1)
    # dataset_type = '(a+b)_mod_97'
    # dataset_type = "Transfer_(a+b)mod97_to_(a+b)mod_p"

    # print(len(dataset), dataset.data.shape, dataset.__getitem__(50))
    # sys.exit(0)



    train_size = int(0.5 * len(dataset))
    if args.aux_loss:
        train_ratio = 0.49
        valid_ratio = 0.5 - train_ratio

        train_size = int(train_ratio * len(dataset))
        valid_size = int(valid_ratio * len(dataset))
        print("VALID SIZE: ", valid_size)
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

    adam_state = {}

    lr = args.lr
    betas = (0.9, 0.98)
    eps = 1e-6


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
    weight_entropy = []
    neuralgrad_entropy = []
    gradient_entropy_after = []
    gradient_entropy_before = []
    weight_norm = []

    lastlayer_attn_out_proj_grad_before = []
    lastlayer_attn_out_proj_grad_after = []

    gradients_before = {name: [] for name, param in model.named_parameters() if "layers" in name}
    gradients_after = {name: [] for name, param in model.named_parameters() if "layers" in name}

    # gradient_entropy_before = {name: [] for name, param in model.named_parameters() if "layers" in name}
    # gradient_entropy_after = {name: [] for name, param in model.named_parameters() if "layers" in name}

    # gradient_entropy_before_avg = {name: [] for name, param in model.named_parameters() if "layers" in name}
    # gradient_entropy_after_avg = {name: [] for name, param in model.named_parameters() if "layers" in name}

    h_t = []

    grads = None
    i = 0

    # For logging network weights.
    net_its, nets = [], []
    total_steps = 0

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
            for idx, input in enumerate(loader):
                input = input.to(device).long().transpose(0,1)

                g_hs_after = []
                g_hs_before = []
        
                if is_train:
                    total_steps += 1

                    if args.neural_grad:
                        
                        cur_innerloop_attn_out_proj_grads_before, cur_innerloop_attn_out_proj_grads_after = [], []

                        with torch.set_grad_enabled(is_train):
                            logits = model(input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            loss = F.cross_entropy(logits[-1], input[-1])
                            total_loss += loss.item() * input.shape[-1]
                        
                        model.zero_grad()
                        loss.backward()

                        
                        g_h_after = 0
                        g_h_before = 0
                        g_h_fg = False

                        for name, param in model.named_parameters():
                            # print(name, param, param.grad)
                            grad = param.grad.view(-1,1)
                            modified_grad = neural_grad(grad)
                            
                            #print("inner_step_idx: ", inner_step_idx ,modified_grad.norm(p=2), grad.norm(p=2))
                            param.grad = modified_grad.view(param.shape)

                            # ## gradient entropy
                            # if inner_step_idx == t - 1:
                            #     g_h_fg = True
                            #     normalized_abs_modified_grad = torch.abs(modified_grad)
                            #     normalized_abs_grad = torch.abs(grad)

                            #     g_h_after = g_h_after - (normalized_abs_modified_grad * torch.log(normalized_abs_modified_grad + 1e-8)).sum()
                            #     g_h_before = g_h_before - (normalized_abs_grad * torch.log(normalized_abs_grad + 1e-8)).sum()



                            ## track the gradients
                            if args.track_grad:
                                if inner_step_idx == t - 1 and "layers" in name:
                                    # gradients_before[name].append(grad.norm(p=2).item())
                                    # gradients_after[name].append(modified_grad.norm(p=2).item())

                                    g_h_fg = True
                                    g_h_after = 0
                                    g_h_before = 0

                                    normalized_abs_modified_grad = torch.abs(modified_grad)# / torch.abs(modified_grad).sum()
                                    normalized_abs_grad = torch.abs(grad)# / torch.abs(grad).sum()

                                    g_h_after = g_h_after - (normalized_abs_modified_grad * torch.log(normalized_abs_modified_grad + 1e-8)).sum()
                                    g_h_before = g_h_before - (normalized_abs_grad * torch.log(normalized_abs_grad + 1e-8)).sum()


                                    gradient_entropy_after[name].append(g_h_after.item())
                                    gradient_entropy_before[name].append(g_h_before.item())



                                    # if f"layers.{args.n_layers-1}.attn.out_proj.weight" in name:
                                    #     cur_innerloop_attn_out_proj_grads_before.append(
                                    #         grad.norm(p=2).item()
                                    #     )
                                    #     cur_innerloop_attn_out_proj_grads_after.append(
                                    #         modified_grad.norm(p=2).item()
                                    #     )
                            
                            
                        if g_h_fg:
                            g_hs_after.append(g_h_after.item())
                            g_hs_before.append(g_h_before.item())
                                
                                    
                        ## suggest to add the norm of the gradient to ensure the normalization in the neural grad    
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()  
                        scheduler.step()

                        
                        ## Outer loop: to use the val_loss to optimize the neural_grad
                        if args.aux_loss and (total_steps % args.t == 0):

                            model_copy = copy.deepcopy(model)
                            adam_state_copy = adam_state.copy()

                           

                            model_copy.zero_grad()
                            neural_grad.zero_grad()

                            ## compute the current training batch loss
                            logits = model_copy(input[:-1])
                            # calculate loss only on the answer part of the equation (last element
                            cur_batch_loss = F.cross_entropy(logits[-1], input[-1])
                            
                            ## precompute the loss to get original gradient
                            cur_batch_loss.backward(retain_graph=False)

                    
                            ## modify the gradients via neural grad
                            for name, module in model_copy.named_modules():
                                # print(name, module)

                                # grad = param.grad.view(-1,1)
                                # modified_grad = neural_grad(grad)

                                if hasattr(module, "weight"):
                                    grad = module.weight.grad.view(-1,1)
                                    
                                    modified_grad = neural_grad(grad).view(module.weight.shape)
                                    # print(modified_grad)
                                    w = module.weight.data
                                    ## first delete
                                    del module.weight
                                    
                                    # ################
                                    # ## then re-write to keep grad_fn
                                    # state = adam_state_copy[name]['weight']
                                    # state['t'] += 1

                                    # ## updating moving average
                                    # state['m'] = betas[0] * state['m'] + (1 - betas[0]) * modified_grad
                                    # state["v"] = betas[1] * state["v"] + (1 - betas[1]) * modified_grad ** 2

                                    # ## compute bias-corrected moving average
                                    # m_hat = state['m'] / (1 - betas[0] ** state['t'])
                                    # v_hat = state['v'] / (1 - betas[1] ** state['t'])

                                


                                    # ## update module
                                    # module.weight = w - lr * m_hat / (torch.sqrt(v_hat + eps))
                                    # ######################

                                    module.weight = w - args.lr * modified_grad.view(w.shape)
                                    # module.weight.requires_grad = True
                                    # print(module.weight)

                                if hasattr(module, "bias") and module.bias is not None:
                                    grad = module.bias.grad.view(-1,1)
                                    modified_grad = neural_grad(grad).view(module.bias.shape)
            


                                    b = module.bias.data
                                
                                    ## first delete
                                    del module.bias
                                    ## then re-wite
                                    
                                    # #####################
                                    # state = adam_state_copy[name]['bias']
                                    # state['t'] += 1

                                    # ## updating moving average
                                    # state['m'] = betas[0] * state['m'] + (1 - betas[0]) * modified_grad
                                    # state['v'] = betas[1] * state['v'] + (1 - betas[1]) * modified_grad ** 2

                                    # ## computing bias-corrected moving average
                                    # m_hat = state['m'] / (1 - betas[0] ** state['t'])
                                    # v_hat = state['v'] / (1 - betas[1] ** state['t'])

                                    # module.bias =  b - lr * m_hat / (torch.sqrt(v_hat + eps))
                                    # #print(module.bias.shape)
                                    # ############################


                                    module.bias = b - args.lr* modified_grad.view(b.shape)
                                
                                if hasattr(module, "in_proj_weight"):
                                    grad = module.in_proj_weight.grad.view(-1,1)
                                    
                                    modified_grad = neural_grad(grad).view(module.in_proj_weight.shape)
                                    # print(modified_grad)
                                    w = module.in_proj_weight.data
                                    ## first delete
                                    del module.in_proj_weight
                                    
                                    # ################
                                    # ## then re-write to keep grad_fn
                                    # state = adam_state_copy[name]['weight']
                                    # state['t'] += 1

                                    # ## updating moving average
                                    # state['m'] = betas[0] * state['m'] + (1 - betas[0]) * modified_grad
                                    # state["v"] = betas[1] * state["v"] + (1 - betas[1]) * modified_grad ** 2

                                    # ## compute bias-corrected moving average
                                    # m_hat = state['m'] / (1 - betas[0] ** state['t'])
                                    # v_hat = state['v'] / (1 - betas[1] ** state['t'])

                                


                                    # ## update module
                                    # module.weight = w - lr * m_hat / (torch.sqrt(v_hat + eps))
                                    # ######################

                                    module.in_proj_weight = w - args.lr * modified_grad.view(w.shape)
                                    # module.weight.requires_grad = True
                                    # print(module.weight)

                                
                                if hasattr(module, "in_proj_bias"):
                                    grad = module.in_proj_bias.grad.view(-1,1)
                                    
                                    modified_grad = neural_grad(grad).view(module.in_proj_bias.shape)
                                    # print(modified_grad)
                                    w = module.in_proj_bias.data
                                    ## first delete
                                    del module.in_proj_bias
                                    
                                    # ################
                                    # ## then re-write to keep grad_fn
                                    # state = adam_state_copy[name]['weight']
                                    # state['t'] += 1

                                    # ## updating moving average
                                    # state['m'] = betas[0] * state['m'] + (1 - betas[0]) * modified_grad
                                    # state["v"] = betas[1] * state["v"] + (1 - betas[1]) * modified_grad ** 2

                                    # ## compute bias-corrected moving average
                                    # m_hat = state['m'] / (1 - betas[0] ** state['t'])
                                    # v_hat = state['v'] / (1 - betas[1] ** state['t'])

                                


                                    # ## update module
                                    # module.weight = w - lr * m_hat / (torch.sqrt(v_hat + eps))
                                    # ######################

                                    module.in_proj_bias = w - args.lr * modified_grad.view(w.shape)
                                
                                    # module.weight.requires_grad = True
                                    # print(module.weight)

                                # print(model)
                                # print(model.token_embeddings.weight)
                                # exit(0)
                                # print(help(model_copy))
                                # print(name, param)
                                
                                # del model_copy._modules[name]
                                # model_copy._modules[name] = param.data - args.lr* modified_grad.view(param.shape)
                                # print(name)
                                # print(model.token_embeddings.weight)
                                # exit(0)
                                # print(model_copy.named_parameters()[name])
                                # sys.exit()
                                ## update the param.weight
                                # param.data = param.data - args.lr* modified_grad.view(param.shape)
                            
                            # new_requires_grad(model_copy, False)
                            # model_copy.eval()

                            final_loss = 0
                            for val_input in valid_loader:
                                val_input = val_input.to(device).long().transpose(0,1)
                                val_logits = model_copy(val_input[:-1])
                                final_loss += F.cross_entropy(val_logits[-1], val_input[-1])
                            
                            final_loss /= len(valid_loader)
                            #final_loss = torch.abs(final_loss - loss.item())

                            print(final_loss)
                            


                            # if args.aux_loss:
                            #     val_loss = 0
                            #     for val_input in valid_loader:
                            #         val_input = val_input.to(device).long().transpose(0,1)
                            #         val_logits = model(val_input[:-1])
                            #         val_loss += F.cross_entropy(val_logits[-1], val_input[-1])
                                
                            #     val_loss /= len(valid_loader)
                        
                            #     final_loss = torch.abs(final_loss - val_loss) + final_loss
                            
                            # print(final_loss)


                            
                            meta_optimizer.zero_grad(set_to_none=True)
                            final_loss.backward(retain_graph=True)
                            
                            # neuralgrad_grad = torch.autograd.grad(
                            #     final_loss,
                            #     neural_grad.parameters(),
                            #     create_graph=True,
                            #     allow_unused=True,
                            # )

                            # print(neuralgrad_grad)


                            # for name, param in neural_grad.named_parameters():
                            #     #print(param)
                            #     print(param.grad)
                            
                            # sys.exit()

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

                        g_h_before = 0
                        g_h_after = 0

                        # for name, param in model.named_parameters():

                        #     grad = param.grad.view(-1,1)

                        #     c = 1/ 1
                        #     ## normalize gradient by l2 norm
                        #     param.grad = c * param.grad.data / ((param.grad.data).norm(2) + 1e-12)

                        #     ## gradient entropy
                        #     normalized_abs_grad = torch.abs(param.grad)
                        #     g_h_before = g_h_before - (normalized_abs_grad * torch.log(normalized_abs_grad + 1e-8)).sum()
                        
                        # g_hs_before.append(g_h_before.item())


                        
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
                            grads, h_t = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger, fft=args.fft, h_t=h_t)



                            # for name, param in model.named_parameters():
                            #     grad = param.grad.view(-1,1)

                            #     ## gradient entropy
                            #     normalized_abs_grad = torch.abs(grad)
                            #     g_h_after = g_h_after - (normalized_abs_grad * torch.log(normalized_abs_grad + 1e-8)).sum()
                            
                            # g_hs_after.append(g_h_after.item())

                            
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


                # ### Weight entropy
                # h = 0
                # for name, param in model.named_parameters():
                #     flatten_param = param.flatten()
                #     h = h - (torch.abs(flatten_param) * torch.log(torch.abs(flatten_param) + 1e-8)).sum()
                #     # for ele in flatten_param:
                #     #     h = h - torch.abs(ele) * torch.log(torch.abs(ele) + 1e-12)
                
                # weight_entropy.append(h.item())
                # ###

                # ### weight norm
                # nm = 0
                # for name, param in model.named_parameters():
                #     flatten_param = param.flatten()
                #     nm = nm + (flatten_param ** 2).sum()
                
                # weight_norm.append(nm.item())
                # ###
                
                # ### gradient entropy
                # if args.neural_grad:
                #     gradient_entropy_after.append(sum(g_hs_after)/len(g_hs_after))
                #     gradient_entropy_before.append(sum(g_hs_before)/len(g_hs_before))
                # elif  args.filter != 'none':
                #     gradient_entropy_after.append(sum(g_hs_after)/len(g_hs_after))
                # else:
                #     gradient_entropy_before.append(sum(g_hs_before)/len(g_hs_before))

                # ### NG entropy
                # h = 0
                # for name, param in neural_grad.named_parameters():
                #     flatten_param = param.flatten()
                #     h = h - (torch.abs(flatten_param) * torch.log(torch.abs(flatten_param + 1e-8))).sum()
                #     # for ele in flatten_param:
                #     #     h = h - torch.abs(ele) * torch.log(torch.abs(ele) + 1e-12)
                
                # neuralgrad_entropy.append(h.item())
                # ###



                if args.track_grad:
                    for name, grad_entropy_list in gradient_entropy_before.items():
                        gradient_entropy_before_avg[name].append(sum(grad_entropy_list) / len(grad_entropy_list))
                    for name, grad_entropy_list in gradient_entropy_after.items():
                        gradient_entropy_after_avg[name].append(sum(grad_entropy_list) / len(grad_entropy_list))

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
            plt.title(f"{dataset_type}", fontsize=15)#_p={args.p} (training on 50% of data)")
            plt.xlabel("Optimization Steps", fontsize=15)
            plt.ylabel("Accuracy", fontsize=15)
            plt.xscale("log", base=10)
            plt.ylim(0,1.05)
            plt.grid()
            if args.neural_grad:
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            
                if not args.tl_eval:
                    torch.save(neural_grad.state_dict(), f"results/acc_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.pt")
            elif args.filter != 'none':
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_filter_{args.filter}.png", dpi=150)
                #torch.save(neural_grad.state_dict(), f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.pt")
            else:
                plt.savefig(f"results/acc_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}.png", dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, test_loss, label="test")
            plt.legend()
            plt.title(f"{dataset_type}", fontsize=15)#_p={args.p}_(training on 50% of data)")
            plt.xlabel("Optimization Steps", fontsize=15)
            plt.ylabel("Loss", fontsize=15)
            plt.xscale("log", base=10)
            plt.grid()
            if args.neural_grad:
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            elif args.filter != "none":
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_filter_{args.filter}.png", dpi=150)
            else:
                plt.savefig(f"results/loss_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}.png", dpi=150)
          
            plt.close()


            # ### Weight entropy
            # plt.plot(steps, weight_entropy)#, label="weight_entropy")
            # #plt.legend()
            # plt.title(f"Weight Entropy")
            # plt.xlabel("Optimization Steps")
            # plt.ylabel("Entropy value")
            # plt.xscale("log", base=10)
            # plt.grid()
            # if args.neural_grad:
            #     plt.savefig(f"results/WeightEntropy_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            # elif args.filter != 'none':
            #     plt.savefig(f"results/WeightEntropy_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_filter_{args.filter}.png", dpi=150)
            # else:
            #     plt.savefig(f"results/WeightEntropy_normed_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}.png", dpi=150)
          
            # plt.close()
            # ####

            # ### Weight entropy
            # plt.plot(steps, weight_norm)#, label="weight_entropy")
            # #plt.legend()
            # plt.title(f"Weight norm")
            # plt.xlabel("Optimization Steps")
            # plt.ylabel("Norm value")
            # plt.xscale("log", base=10)
            # plt.grid()
            # if args.neural_grad:
            #     plt.savefig(f"results/WeightNorm_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            # elif args.filter != 'none':
            #     plt.savefig(f"results/WeightNorm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_filter_{args.filter}.png", dpi=150)
            # else:
            #     plt.savefig(f"results/WeightNorm_normed_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}.png", dpi=150)
          
            # plt.close()
            # ####

            # ### Gradient entropy
            # if args.neural_grad:
            #     plt.plot(steps, gradient_entropy_after, label="modified gradient entropy")
            #     plt.plot(steps, gradient_entropy_before, label="original gradient entropy")
            # elif args.filter != 'none':
            #     plt.plot(steps, gradient_entropy_after, label="modified gradient entropy")
            # else:
            #     plt.plot(steps, gradient_entropy_before, label="original gradient entropy")

            # plt.legend()
            # plt.title(f"Gradient Entropy")
            # plt.xlabel("Optimization Steps")
            # plt.ylabel("Entropy value")
            # plt.xscale("log", base=10)
            # plt.grid()
            # if args.neural_grad:
            #     plt.savefig(f"results/GradientEntropy_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            # elif args.filter != "none":
            #     plt.savefig(f"results/GradientEntropy_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}_filter_{args.filter}.png", dpi=150)
            # else:
            #     plt.savefig(f"results/GradientEntropy_Nonormed_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_wd{args.weight_decay}.png", dpi=150)
          
            # plt.close()
            ####

            # ### NG entropy
            # plt.plot(steps, neuralgrad_entropy)#, label="weight_entropy")
            # #plt.legend()
            # plt.title(f"NeuralGrad Weight Entropy")
            # plt.xlabel("Optimization Steps")
            # plt.ylabel("Entropy value")
            # plt.xscale("log", base=10)
            # plt.grid()
            # plt.savefig(f"results/NGWeightEntropy_{dataset_type}_p={args.p}_AuxLoss_{args.aux_loss}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=150)
            
            # plt.close()
            # ####

            if args.track_grad:
                n_params = len(gradient_entropy_before_avg)
                cols = 4
                rows = math.ceil(n_params / cols)

                if args.neural_grad:
                    fig, axes = plt.subplots(rows, cols, figsize=(24, 6*rows))
                    axes = axes.flatten()

                    for idx, (name, grads_list) in enumerate(gradient_entropy_after_avg.items()):
                        grads_list_before = gradient_entropy_before_avg[name]

                        axes[idx].plot(grads_list, label="modified")
                        axes[idx].plot(grads_list_before, label="original")
                        axes[idx].set_title(f"{name}", fontsize=15)
                        axes[idx].set_xlabel("Optimization Steps", fontsize=15)
                        axes[idx].set_ylabel("Entropy value")
                        axes[idx].set_xscale('log', base=10)
                        axes[idx].grid(True)
                        
                        if idx == 0:
                            ## set legend
                            axes[idx].legend(loc='upper right', fontsize=15)
                    
                    for ax in axes[n_params:]:
                        ax.axis("off")
                    
                    plt.savefig(f"results/grad_entropy_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.t}InnerLoop.png", dpi=250)
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
                    
                    plt.savefig(f"results_new/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_filter_{args.filter}.png", dpi=250)
                    plt.close()

            if args.fft:
                h_t2 = np.array(h_t)
                freq_domain = np.fft.fft(h_t2)
                freqs = np.fft.fftfreq(len(h_t2))
                # print(freqs)

                plt.plot(freqs[:len(freqs)//2], np.abs(freq_domain)[:len(freq_domain)//2])

                plt.title("Frequency domain", fontsize=16)
                plt.xlabel("Frequency", fontsize=14)
                plt.ylabel("Amplitude", fontsize=14)
                plt.tight_layout()
                plt.grid()
                #plt.savefig("frequency_domain_vanilla_training_originalgrad_dim200.png", dpi=250)
                plt.savefig("frequency_domain_NeuralGrad_training_addedgrad_dim100.png", dpi=250)
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
                #     plt.savefig(f"results/grad_norm_{dataset_type}_p={args.p}_TD_{args.n_layers}Layers_{args.n_heads}Heads_{args.dim}Dim_lr{args.lr}_NeuralGrad_{args.neural_layers}NeuralLayers_{args.neural_alpha}Alpha_{args.neural_beta}Beta_{args.t}InnerLoop.png", dpi=150)
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
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--neural_hidden_dim", type=int, default=32)
    parser.add_argument("--neural_layers", type=int, default=3)
    parser.add_argument("--neural_alpha", type=int, default=16)
    parser.add_argument("--neural_grad", action='store_true')
    parser.add_argument("--tl_eval", action="store_true")
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--track_grad", action="store_true")
    parser.add_argument("--fft", action="store_true")

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
