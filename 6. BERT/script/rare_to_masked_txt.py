import sys
sys.path.insert(0,'..')

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from src.util import callpath
from src.preprocess import create_instances_from_lines

## Settings
num_files = 20
simple_length = 128
max_length = 128
file_iter = 4

## Load Tokenizer
tokenizer = Tokenizer(WordPiece()).from_file(callpath("data/tokenizer.json"))


timechecker = tqdm(total=num_files * file_iter)

instances = 0

for iter_i in range(file_iter):
    # length = 512 at first iter, 128 in the others.
    #if iter_i in (1,2,3):
    #    timechecker.update(num_files)
    #    continue
    
    length = simple_length if iter_i else max_length
    
    for file_i in range(1, num_files+1):
        with open(callpath(f"data/rare/{file_i}.txt"), 'r') as f:
            lines = f.readlines()
    
        instances += create_instances_from_lines(file_lines=lines,
            out_file=callpath(f"data/done/{iter_i}_th_{file_i}.json"),
            min_term=5000, max_seq_length=length, short_seq_prob=.1,
            masked_lm_prob=.15, max_predictions_per_seq=int(.25*length), tokenizer=tokenizer)

        timechecker.set_description(f"total {instances} sentences")
        timechecker.update(1)