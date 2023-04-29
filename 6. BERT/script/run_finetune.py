import sys, os
sys.path.insert(0, "../")

from src.model import BERT_LM
from src.trainer import Tuner
from src.config import Config_mini
from src.util import callpath, terminal_bool

import torch
import argparse

parser = argparse.ArgumentParser(description="----BERT-Pre-Train----")

parser.add_argument("--task", default="sst2", type=str, help="task name for finetuning")
parser.add_argument("--device", default='cuda:1', type=str, choices = ['cpu', 'cuda:0', 'cuda:1'], help="device where training is executed")
parser.add_argument("--is_pre_norm", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to lay a Norm on : default) False")

parsed_opt = parser.parse_args()
config = Config_mini()

config.set_task(parsed_opt.task)
config.is_pre_norm = terminal_bool(parsed_opt.is_pre_norm)
config.device = parsed_opt.device

config.use_tqdm = False
config.task_dir = [callpath(filename) for filename in config.task_dir]

model = BERT_LM(config)
sign = "pre" if config.is_pre_norm else 'post'

best_score = 0

for b in (8, 16, 32, 64, 128):
    for lr in (3e-4, 1e-4, 5e-5, 3e-5):
        config.task_lr = lr
        config.task_batch = b
        
        #print(f"lr : {config.task_lr} / batch : {config.task_batch}")
        
        with open(callpath(f"result/tiny_{sign}.pt"), 'rb') as f:
            load = torch.load(f, map_location = config.device)
            model.load_state_dict(load['model'])
            del load
            
        trainer = Tuner(model.BERT, config)
        trainer.tune()
        score = trainer.test()
        
        if score > best_score:
            best_score = score
            best_param = (lr, b)
            
print(f"Best Score is {best_score*100:.2f}\nAt LR = {best_param[0]}, batch = {best_param[1]}")