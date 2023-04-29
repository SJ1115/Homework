import os
import torch.nn as nn

# for Trainer.py
def src_to_lines(src):
    with open(src, 'r', encoding='utf8') as f:
        lines = f.readlines()
    return lines

# for Model.py
def init_weight(Module, att = False):
    if isinstance(Module, nn.Linear):
        gain = .63 if att else 1
        nn.init.xavier_normal_(Module.weight, gain)
        
        if Module.bias is not None:
            nn.init.constant_(Module.bias, 0)
    
    elif isinstance(Module, nn.Embedding):
        nn.init.normal_(Module.weight, mean=0, std=512**-.5)

# for run.py
def callpath(filename):
    return os.path.join(os.path.dirname(__file__), '..', filename)

def terminal_bool(arg):
    arg = arg.lower()
    return arg in ('t', 'y')