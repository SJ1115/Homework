import sys, os

sys.path.insert(0,'..')

import argparse
import pickle
import numpy as np
from sklearn.utils import shuffle

from src.model import TextClassifier
from src.preprocess import huffman, corpus_bigram


def bigram_bool(arg):
    arg = arg.lower()
    return arg in ('t', 'y')

parser = argparse.ArgumentParser(description="-----[fastText Text.Classifier]-----")
parser.add_argument("--data", default="AG", help="available datasets: AG, SOGOU, DBPEDIA, YAHOO, AMAZON_P, AMAZON_F, YELP_P, YELP_F")
#parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
parser.add_argument("--bigram", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to build vocab with bigram")
parser.add_argument("--epoch", default=5, type=int, help="number of max epoch")
parser.add_argument("--learning_rate", nargs='+', default=[.05, .1, .25, .5], type=float, help="sequence of learning rate")


options = parser.parse_args()

data = options.data.lower()
epoch_size = options.epoch
bigram = bigram_bool(options.bigram)

lr_lst = options.learning_rate

with open(os.path.join(os.path.dirname(__file__), '..', "script/" + data + ".pkl"), 'rb') as f:
    data_out = pickle.load(f)


### re-construct Corpus, for Bigram Computation ###
w2i, train, train_label, test, test_label = corpus_bigram(data_out, bigram)

### Build Huffman tree for Hierarchical Softmax(Output) ##

huff = huffman(corpus = train_label)

huff.build_codebook()

### Training ###

score = 0
lr_max = None
print(f"Training for {data.upper()}, with {'Bi' if bigram else 'Uni'}-gram, at", end='')

for learn_rate in lr_lst:
    model = TextClassifier(len(w2i), huff, hidden = 10)

    print(f" {learn_rate} ", end='')
    lr = learn_rate
    count=0
    
    for epoch in range(epoch_size):
        train_x, train_y = shuffle(train, train_label)
        for i, (x, y) in enumerate(zip(train_x, train_y)):
            model.learn(x, y, lr)

            count += 1

            lr = learn_rate * max(.001, (1 - count/(epoch_size*len(train_label))))
                
    scoretemp = model.test(test, test_label)
    if scoretemp > score:
        score = scoretemp
        lr_max = learn_rate

print(f": Best Score is {score*100:.2f} at {lr_max}")