import sys, os

sys.path.insert(0,'..')

import argparse
import pickle

from src.preprocess import preprocess
from src.load import url



def bigram_bool(arg):
    arg = arg.lower()
    return arg in ('t', 'y')

parser = argparse.ArgumentParser(description="-----[fastText Text.Classifier]-----")
parser.add_argument("--reverse", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to build vocab with bigram")
parser.add_argument("--unk", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to build vocab with bigram")
parser.add_argument("--vocab_size", default=50000, type=int, help="vocab size")

options = parser.parse_args()

unk = bigram_bool(options.unk)
reverse = bigram_bool(options.reverse)
vocab = options.vocab_size



### preprocess ###

eng_to_idx, ger_to_idx, eng_train, ger_train, eng_dev, ger_dev, eng_test, ger_test = preprocess(url, max_vocab=vocab, reverse=reverse, get_unk=unk)

dataset = {}
dataset['eng_to_idx'] = eng_to_idx
dataset['ger_to_idx'] = ger_to_idx
dataset['eng_train'] = eng_train
dataset['eng_test'] = eng_test
dataset['eng_dev'] = eng_dev
dataset['ger_train'] = ger_train
dataset['ger_test'] = ger_test
dataset['ger_dev'] = ger_dev

### save ###

with open(os.path.join(os.path.dirname(__file__), '..', "data/" + f"unk({unk})_rev({reverse})" + ".pkl"), 'wb') as f:
    pickle.dump(dataset, f)