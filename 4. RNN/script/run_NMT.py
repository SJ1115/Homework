import sys, os

sys.path.insert(0,'..')

import torch
import pickle

from src.model import NMT, Baseline
from src.trainer import Trainer
from src.func import args_bool, dtype_choice, false_check

import argparse

###### Parsing Arguments ######
parser = argparse.ArgumentParser(description="-----[fastText Text.Classifier]-----")
    # Setting from preprocessing
parser.add_argument("--reverse", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to reverse in-sequence")
parser.add_argument("--unk", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to build vocab with <unk> token")
    # Setting for basic submodel : embedding and LSTM
parser.add_argument("--embedding", default=1000, type=int, help="dimension size of word Embedding")
parser.add_argument("--hidden", default=1000, type=int, help="dimension size of LSTM hidden")
parser.add_argument("--depth", default=4, type=int, help="depth of LSTM layers")
parser.add_argument("--dropout", default=.2, type=float, help="dropout rate of LSTM")
    # Setting for Attention
parser.add_argument("--attention", default='global', type=str, choices=['g', 'lp', 'lm', 'loc_pred', 'loc_mono', 'global', 'local_predictive', 'local_monotonic', 'False', 'false', 'f'], help="whether to build vocab with <unk> token")
parser.add_argument("--scoring", default='gen', type=str, choices=['d', 'g', 'c', 'l', 'dot', 'gen', 'con', 'loc', 'general', 'concat', 'location', 'False', 'false', 'f'], help="whether to build vocab with <unk> token")
parser.add_argument("--in_feed", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to use input feeding approach")
parser.add_argument("--window", default=10, type=int, help="window size in local attention")
    # Setting for Training
parser.add_argument("--lr", default=1, type=float, help="learning rate")
parser.add_argument("--epoch", default=12, type=int, help="total epoch num")
parser.add_argument("--decreasing_point", default=8, type=int, help="LR-decreasing point epoch for LR scheduling")
parser.add_argument("--patience", default=3, type=int, help="patience count for early stopping")
    # Setting for runnability
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--dtype", default='float', type=str, choices = ['half', 'float'], help="dtype of model parameter")
parser.add_argument("--device", default='cuda:0', type=str, choices = ['cpu', 'cuda:0', 'cuda:1'], help="device which training executed")
parser.add_argument("--out_name", default='sample.pt', type=str, help="filename.pt for result model")

options = parser.parse_args()

###### SETTING Parameters #######
    # from preprocessing
reverse = args_bool(options.reverse)
unk = args_bool(options.unk)
    # embedding, lstm
embedding = options.embedding
hidden_size = options.hidden
depth = options.depth
dropout = options.dropout
    # Attention
attention = false_check(options.attention), false_check(options.scoring)
attention = None if attention == (None, None) else attention
input_feed = args_bool(options.in_feed)
window_size = options.window
    # Training
learn_rate = options.lr
epoch_size = options.epoch
decreasing = options.decreasing_point
patience = options.patience
    # runnability
batch = options.batch_size
dtype = dtype_choice(options.dtype)
device = options.device
out_name = options.out_name
###### Getting Data ########

with open(os.path.join(os.path.dirname(__file__), '..', "data/" + f"unk({unk})_rev({reverse})" + ".pkl"), 'rb') as f:
    dataset = pickle.load(f,)

eng_to_idx = dataset['eng_to_idx']
ger_to_idx = dataset['ger_to_idx']
eng_train = dataset['eng_train']
eng_test = dataset['eng_test']
eng_dev = dataset['eng_dev']
ger_train = dataset['ger_train']
ger_test = dataset['ger_test']
ger_dev = dataset['ger_dev']

id_to_ger = {}
for w in ger_to_idx:
    id_to_ger[ger_to_idx[w]] = w

###### define & run Model ######
model = NMT(attention=attention, enc_vocab_size=len(eng_to_idx), dec_vocab_size=len(ger_to_idx), embedding_dim=embedding, hidden_size=hidden_size, depth=depth, dropout=dropout, input_feed=input_feed, reverse=reverse, window_size=window_size, length=52)
#model = Baseline(enc_vocab_size=len(eng_to_idx), dec_vocab_size=len(ger_to_idx), embedding_dim=embedding, hidden_size=hidden_size, depth=depth, dropout=dropout, reverse=reverse)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate) # , rho=.95, weight_decay=1e-8

data = (eng_train, ger_train, eng_test, ger_test)
trainer = Trainer(model, criterion, optimizer, id_to_ger, data, batch, device)

trainer.verbose(True)
trainer.train(
    epoch=epoch_size, 
    filename=os.path.join(sys.path[0], "result/" + out_name),
    early_stopping=patience, decreasing_point=decreasing)

###### test & save Model ######
p, b = trainer.test()
print(f"PPL : {p:.4f}, BLEU : {b:.4f}")