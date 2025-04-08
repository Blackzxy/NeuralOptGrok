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

import inspect
    
def train(args,
          inner_loader,
          outer_loader,
          test_loader,
          model,
          amp = None,
          device = "cuda",
          inner_loop_steps = 1,
          wandb_report = True):
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    if amp is not None:
        nparams_amp = sum([p.numel() for p in amp.parameters() if p.requires_grad])
        print(f'num. params in amplifier: [{nparams_amp}]')
    print(f'num. params in base model: [{nparams/1e6}M]')
    
    
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    if args.neuralgrok:
        meta_optimizer = getattr(torch.optim, args.optimizer)(
                amp.parameters(),
                lr=1e-4,
                weight_decay = args.weight_decay,
                betas=(args.beta1, args.beta2),
            )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    
    steps_per_epoch = len(inner_loader)
    num_epochs = int(args.budget) // steps_per_epoch
    print(f"Num. of epochs: [{num_epochs}]")

    its, eval_its, train_acc, test_acc, train_loss, test_loss = [], [], [], [], [], []

    weight_norms, weight_entropys = [], []    
    grad_norms, grad_entropys = [], []
    
    it = 0
    for e in tqdm(range(num_epochs)):
        wandb_logs = {}
        # train
        model.train()
        epoch_train_loss, epoch_train_acc = 0, 0
        for input in inner_loader:
            it += 1
            input = input.to(device).long().transpose(0,1)
            
            # if amp is not None and (it+2) % inner_loop_steps == 0:
            #     model_copy = copy.deepcopy(model)
            #     model_copy.zero_grad()
                    
            with torch.set_grad_enabled(True):
                logits = model(input[:-1])
                # calculate loss only on the answer part of the equation (last element
                loss = F.cross_entropy(logits[-1], input[-1])

            model.zero_grad()
            loss.backward()
            
            acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
            epoch_train_acc += acc.item()
            epoch_train_loss += loss.item()
            
            if args.neuralgrok:
                transform_grads(model, amp, is_inner=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if it<10 or (it<1000 and it%64==0) or (it>1000 and it%500==0) == 0:
                w_norm, w_entropy, g_norm, g_entropy = get_entropy(model)
                eval_its.append(it)
                weight_norms.append(w_norm)
                weight_entropys.append(w_entropy)
                grad_norms.append(g_norm)
                grad_entropys.append(g_entropy)
                wandb.log({
                    "global_step": it,
                    "weight_norm": w_norm,
                    "weight_entropy": w_entropy,
                    "grad_norm": g_norm,
                    "grad_entropy": g_entropy,
                }, commit=True)
            
            
            optimizer.step()  
            scheduler.step()
            
            if args.neuralgrok and (it % inner_loop_steps == 0):
                model_copy = copy.deepcopy(model)
                model_copy.zero_grad()
                amp_update(args, amp, meta_optimizer, outer_loader, model_copy, inner_batch=input, device=args.device)
               
            # it += 1
        epoch_train_loss /= len(inner_loader)
        epoch_train_acc /= len(inner_loader)
        
        # test
        epoch_test_loss, epoch_test_acc = eval_model(test_loader, model, device=args.device)
        
        its.append(it)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        wandb.log({
            "global_step": it,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc,
        }, commit = True)

def eval_model(test_loader, model, device):
    epoch_test_loss, epoch_test_acc = 0, 0

    with torch.set_grad_enabled(False):
        for input in test_loader:
            input = input.to(device).long().transpose(0,1)
            logits = model(input[:-1])
            # calculate loss only on the answer part of the equation (last element
            loss = F.cross_entropy(logits[-1], input[-1])
            acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
            epoch_test_acc += acc.item()
            epoch_test_loss += loss.item()
        epoch_test_loss /= len(test_loader)
        epoch_test_acc /= len(test_loader)
    return epoch_test_loss, epoch_test_acc

def get_metric(w):
    norm = w.norm().item()
    entropy = -torch.sum(torch.abs(w)*torch.log(torch.abs(w))).item()
    return norm, entropy

def get_entropy(model):
    flat_w, flat_g = [], []
    for name, param in model.named_parameters():
        flat_w.append(param.flatten())
        if not param.requires_grad:
            continue
        flat_g.append(param.grad.flatten())
    
    flat_w = torch.concat(flat_w)
    flat_g = torch.concat(flat_g)
    w_norm, w_entropy = get_metric(flat_w)
    g_norm, g_entropy = get_metric(flat_g)
    return w_norm, w_entropy, g_norm, g_entropy
    
def transform_grads(model, amp, is_inner=True):
    # if is_inner:
    #     amp.eval()
    # else:
    #     amp.train()
    
    trans_grads = {}
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        #     continue
        grad = param.grad.view(-1,1)
        grad_trans = amp(grad)
        param.grad = grad_trans.view(param.shape)
        trans_grads[name] = grad_trans.view(param.shape)
    return trans_grads 

def trans_module_weights(model_copy, amp, lr=1e-3):

    for name, module in model_copy.named_modules():
        if hasattr(module, "weight"):
            grad = module.weight.grad.view(-1,1)
            
            modified_grad = amp(grad).view(module.weight.grad.shape)
            # print(modified_grad)
            w = module.weight.data
            ## first delete
            del module.weight

            module.weight = w - lr * modified_grad.view(w.shape)

        if hasattr(module, "bias") and module.bias is not None:
            grad = module.bias.grad.view(-1,1)
            modified_grad = amp(grad).view(module.bias.grad.shape)
            b = module.bias.data
        
            ## first delete
            del module.bias
            ## then re-wite
            module.bias = b - lr* modified_grad.view(b.shape)
        
        if hasattr(module, "in_proj_weight"):
            grad = module.in_proj_weight.grad.view(-1,1)
            
            modified_grad = amp(grad).view(module.in_proj_weight.grad.shape)
            # print(modified_grad)
            w = module.in_proj_weight.data
            ## first delete
            del module.in_proj_weight

            module.in_proj_weight = w - lr * modified_grad.view(w.shape)

        if hasattr(module, "in_proj_bias"):
            grad = module.in_proj_bias.grad.view(-1,1)
            
            modified_grad = amp(grad).view(module.in_proj_bias.grad.shape)
            # print(modified_grad)
            w = module.in_proj_bias.data
            ## first delete
            del module.in_proj_bias
            module.in_proj_bias = w - lr * modified_grad.view(w.shape)
    # for name, params in module.named_parameters():
    #     if hasattr(params, "grad"):
            
    #         params = params.data
    #     print(name)
    #     print(hasattr(params, "weight"))
    #     import pdb
    #     pdb.set_trace()
    
def amp_update(args, amp, meta_opt, outer_loader, model_copy, inner_batch, device):
    amp.zero_grad()
    
    logits_cp = model_copy(inner_batch[:-1])
    loss_cp = F.cross_entropy(logits_cp[-1], inner_batch[-1])
    loss_cp.backward(retain_graph=False)
    
    trans_module_weights(model_copy, amp, lr=args.lr)
    
    
    outer_loss = 0
    for input in outer_loader:
        input = input.to(device).long().transpose(0,1)
        logits = model_copy(input[:-1])
        outer_loss += F.cross_entropy(logits[-1], input[-1])
    
    outer_loss /= len(outer_loader)

    meta_opt.zero_grad(set_to_none=True)
    outer_loss.backward(retain_graph=True)
    #check_amp_grads(amp) ## check if the gradients are None
    meta_opt.step()
    return outer_loss.item()
    
def check_amp_grads(amp):
    for name, param in amp.named_parameters():
        # print(param.grad)
        # sys.exit()
        # import pdb
        # pdb.set_trace()
        assert param.grad is not None, "Outer-loop fails!"


