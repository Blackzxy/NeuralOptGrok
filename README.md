# NeuralOptGrok
This code repo is for the paper [NeuralGrok:
Accelerate Grokking by Neural Gradient Transformation](https://arxiv.org/pdf/2504.17243)
## Usage

You can run the experiments in the following codes (Please don't forget to previde your personal WANDB API keys):
```bash
bash scripts/neuralgrok/task1.sh
```


For the model *NeuralGrok*, it is defined in `src/model.py`. 

Note that in different tasks, you may need to change the `num_tokens` in `src/run.py` Line 80 accordingly:
```
a+b, a-b, axb: num_tokens=args.p + 2;
axa-b: num_tokens=args.p + 3;
axc+bxd-e: num_tokens=args.p + 4;
```



