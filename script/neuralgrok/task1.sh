#!/bin/bash
cd /scratch/homes/sfan/NeuralOptGrok
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"

python src/run.py --task_type a+b --p 97 --neuralgrok
# python src/run.py --task_type a-b --p 97 --neuralgrok
# python src/run.py --task_type axb --p 97 --neuralgrok
# python src/run.py --task_type axa-b --p 97 --neuralgrok
# python src/run.py --task_type axc+bxd-e --p 7 --neuralgrok


# python src/run.py --task_type a+b --p 97 
# python src/run.py --task_type a-b --p 97 
# python src/run.py --task_type axb --p 97 
# python src/run.py --task_type axa-b --p 97 
# python src/run.py --task_type axc+bxd-e --p 7 