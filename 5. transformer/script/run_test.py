import sys, os
sys.path.insert(0,'..')

import torch
import dill
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
import argparse

from src.Trainer import Trainer
from src.Func import callpath, terminal_bool

parser = argparse.ArgumentParser(description="-----[\"Test for transformer\" Seq2Seq Translator]-----")
    # Setting for Files : Path
parser.add_argument("--lang", default="de", type=str, choices=['de','fr'], help="language for the model")
parser.add_argument("--filename", default="sample.pt", type=str, help="filename for model.pt")
parser.add_argument("--device", default="cpu", type=str, help="device to run test")
parser.add_argument("--beam", default="T", type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to use beam search")
parser.add_argument("--verbose", default="T", type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to show progress")
parser.add_argument("--dev", default="F", type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to show progress")
parser.add_argument("--max_len", default=128, type=int, help="max length in 'PREDICT' fucntion")
parser.add_argument("--margin_len", default=50, type=int, help="max length in 'PREDICT' fucntion")

options = parser.parse_args()
lang_out = options.lang
filename = callpath(f"result/{options.filename}")
device = options.device
beam = terminal_bool(options.beam)
verbose = terminal_bool(options.verbose)
max_len = options.max_len; margin_len = options.margin_len
set_ = 'val' if terminal_bool(options.verbose) else 'test'

param = torch.load(filename, map_location=device, pickle_module=dill)
print(f"{options.filename} has been trained {param['step']}\'th steps")
model = param['model']
tokenizer = param['tokenizer']

#
src=(callpath(f"data/en_{lang_out}/{set_}_en.txt"), callpath(f"data/en_{lang_out}/{set_}_{lang_out}.txt"))

trainer = Trainer(tokenizer=tokenizer, model=model, criterion=None, batch_size=16, device=device)
trainer.set_verbose(verbose)
b = trainer.test(src=src, beam=beam,
    max_len=max_len, margin_len=margin_len,num_workers=0
)

print(f"BLEU score : {100*b:.2f}")