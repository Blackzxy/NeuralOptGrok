#!/bin/bash
# cd /scratch/homes/sfan/NeuralOptGrok
# pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="YOUR_API_KEY"

# python src/run.py --task_type axc+bxd-e --p 7 --lr 1e-3 --weight_decay 1e-3 --dim 128 --n_layers 4 --n_heads 4  --neuralgrok  --neural_hidden_dims 1152 --inner_loop_steps 1
python src/run.py --task_type a+b --p 97 --lr 1e-3 --weight_decay 1e-3 --dim 128 --n_layers 2 --n_heads 4  --neuralgrok  --neural_hidden_dims 128,128 --inner_loop_steps 4
# python src/run.py --task_type a-b --p 97 --lr 1e-3 --weight_decay 1e-3 --dim 128 --n_layers 2 --n_heads 4  --neuralgrok  --neural_hidden_dims 128,128 --inner_loop_steps 4
# python src/run.py --task_type axb --p 97 --lr 1e-3 --weight_decay 1e-3 --dim 128 --n_layers 2 --n_heads 4  --neuralgrok  --neural_hidden_dims 128,128 --inner_loop_steps 4
# python src/run.py --task_type axa-b --p 97 --lr 1e-3 --weight_decay 1e-3 --dim 128 --n_layers 2 --n_heads 4  --neuralgrok  --neural_hidden_dims 128,128 --inner_loop_steps 4



# python src/run.py --task_type a+b --p 97 
# python src/run.py --task_type a-b --p 97 
# python src/run.py --task_type axb --p 97 
# python src/run.py --task_type axa-b --p 97 
# python src/run.py --task_type axc+bxd-e --p 7 