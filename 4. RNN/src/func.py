import torch
import torch.nn as nn
#### for attention.py ####

def score_check(mode):
    if not mode:
        return None
    mode = mode.lower()
    if mode not in ('d', 'g', 'c', 'l', 'dot', 'gen', 'con', 'loc', 'general', 'concat', 'location'):
        ValueError("\"Mode\" is not Valid.")
    if mode in ('d', 'dot'):
        return 'dot'
    elif mode in ('g', 'gen', 'general'):
        return 'gen'
    elif mode in ('c', 'con', 'concat'):
        return 'con'
    elif mode in ('l', 'loc', 'location'):
        return 'loc'

#### for layer.py ####

def align_check(mode):
    if not mode:
        return None
    mode = mode.lower()
    if mode not in ('g', 'lp', 'lm', 'loc_pred' 'loc_mono', 'global', 'local_predictive', 'local_monotonic'):
        ValueError("\"Mode\" is not Valid.")
    if mode in ('lp', 'loc_pred', 'local_predictive'):
        return 'loc_pred'
    elif mode in ('lm', 'loc_mono', 'local_monotonic'):
        return 'loc_mono'
    elif mode in ('g', 'global'):
        return 'global'

def init_lstm(model):
    for layer in model.all_weights:
        for param in layer:
            if param.ndim == 2:
                nn.init.uniform_(param, -.1, .1)
            else:
                nn.init.constant_(param, 0)

#### for model.py ####

def prop_mode(input_feed=False, train=True):
    return 2*input_feed + (not train)


def last_hidden(hs):
    # only the last hidden state for encoder
    hs = [
        hs[0][:, -1, :].contiguous(),
        hs[1][:, -1, :].contiguous()
    ]
    return hs
#### for run_NMT.py ####

def args_bool(arg):
    arg = arg.lower()
    return arg in ('t', 'y')
def dtype_choice(arg):
    if arg == 'half':
        return torch.half
    elif arg == 'float':
        return torch.float
def false_check(arg):
    arg = arg.lower()
    if arg in ('false', 'f'):
        return None
    else:
        return arg

