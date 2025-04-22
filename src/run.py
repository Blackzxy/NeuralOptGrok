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
import wandb

from dataset import get_dataset, get_dataloader
from model import NeuralAmplifier, Decoder
from train import train

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_proj', default='NeuralGrok', type=str)
parser.add_argument('--wandb_run', default=None, type=str)

parser.add_argument("--label", default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p", type=int, default=97)
parser.add_argument("--budget", type=int, default=3e5)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.98)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--optimizer", default="Adam")
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--dim", type=int,default=128)
parser.add_argument("--n_layers",type=int, default=2)
parser.add_argument("--interval", type=int, default=1000)

parser.add_argument("--inner_loop_steps", type=int, default=2)
parser.add_argument("--neural_hidden_dims", type=str, default="64,128,64")
parser.add_argument("--neuralgrok", action='store_true')
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--task_type", type=str, default="a+b")
    
def get_run_name(args):
    base_name = args.wandb_run
    if base_name is None:
        base_name = ""
    if args.neuralgrok:
        base_name += "NeuralGrok" 
    else:
        base_name += "BASE" 
    base_name += f"-task[{args.task_type}]"
    base_name += f"-wd[{args.weight_decay}]"
    if args.neuralgrok:
        base_name += f"-h[{args.neural_hidden_dims}]"
        base_name += f"-T[{args.inner_loop_steps}]"
    return base_name

def run(args):
    wandb_run_name = get_run_name(args)
    wandb.init(
        project=args.wandb_proj,
        name=wandb_run_name,
    )
    
    
    # get data
    print(f"Load task dataset for [{args.task_type} mod {args.p}]...")
    dataset, seq_len = get_dataset(args.task_type, p=args.p)
 
    print("Seq_len = ", seq_len)

    torch.manual_seed(args.seed)

    model = Decoder(
        dim=args.dim, num_layers=args.n_layers, num_heads=args.n_heads, 
        num_tokens=args.p + 2, seq_len=seq_len,
    ).to(args.device)

    print(model)
    
    amp = None
    if args.neuralgrok:
        
        hidden_dims = [int(h) for h in args.neural_hidden_dims.split(",")]
        amp = NeuralAmplifier(
            input_dim=1,
            hidden_dims=hidden_dims,
            output_dim=1,
        ).to(args.device)


        print(amp)
        # for name, param in amp.named_parameters():
        #     print(name, param)
        # sys.exit()
    
    if args.neuralgrok:
        inner_loader, outer_loader, test_loader = get_dataloader(dataset, p_train=0.5, p_outer=0.01)
    else:
        inner_loader, outer_loader, test_loader = get_dataloader(dataset, p_train=0.5, p_outer=0)
    
    # start training
    print("== Training Starts 🧨 ==")
    train(args,
          inner_loader,
          outer_loader,
          test_loader,
          model,
          amp = amp,
          device = args.device,
          inner_loop_steps = args.inner_loop_steps,
          wandb_report = True)
    print("== Training Finished! 🎏 ==")

if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    
    args = parser.parse_args()
    run(args)
    
    
    
    
    
    
    
    
    