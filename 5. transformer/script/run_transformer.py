import sys, os
sys.path.insert(0,'..')

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
import argparse

from src.Model import Transformer
from src.Trainer import Trainer
from src.Func import callpath, terminal_bool


parser = argparse.ArgumentParser(description="-----[\"transformer\" Seq2Seq Translator]-----")
    # Setting for Files : Path
parser.add_argument("--tokenizer", default="tokenizer.json", type=str, help="filename for tokenizer.json")
parser.add_argument("--out_lang", default="de", type=str, choices=['de', 'fr'],help="output(to) language")
parser.add_argument("--out_model", default="sample.pt", type=str, help="filename for output model.pt")
    # Setting for Model : Transformer
parser.add_argument("--max_len", default=128, type=int, help="maximum length of input for Transformer")
parser.add_argument("--num_layers", default=6, type=int, help="number of Self-Attention Layer")
parser.add_argument("--num_head", default=8, type=int, help="number of heads in Multi-Head Self-Attention")
parser.add_argument("--model_dim", default=512, type=int, help="dimension of each sublayer in entire Model")
parser.add_argument("--value_dim", default=64, type=int, help="dimension of value in Self-Attention")
parser.add_argument("--key_dim", default=64, type=int, help="dimension of key(query) in Self-Attention")
parser.add_argument("--feed_dim", default=2048, type=int, help="dimension in Position Wise Feeed Forward Layer")
parser.add_argument("--dropout", default=.1, type=float, help="Dropout Rate")
parser.add_argument("--send_each", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to send each output in Enc-Dec transaction")
parser.add_argument("--embed_share", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to use Embedding Sharing")
parser.add_argument("--scale_embedding", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to use Scaling in Embedding Layer")
    # Setting for Training
parser.add_argument("--label_smooth", default=.1, type=float, help="Label Smoothing rate")
parser.add_argument("--lr", default=2, type=float, help="Learning rate")
parser.add_argument("--total_step", default=100000, type=int, help="total step(s) for training")
parser.add_argument("--mini_batch", default=32, type=int, help="batch size for device")
parser.add_argument("--total_batch", default=640, type=int, help="batch size for calculation")
parser.add_argument("--warmup", default=4000, type=int, help="warm-up steps")
    # Setting for Runnability
parser.add_argument("--num_workers", default=0, type=int, help="num_workers in torch.DataLoader")
parser.add_argument("--device", default='cuda:0', type=str, choices = ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help="device where training is executed")
parser.add_argument("--verbose", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to show progress bar")

options = parser.parse_args()
    # Setting for File Path
token_file = options.tokenizer
lang_in = 'en'
lang_out = options.out_lang
num_txt = 4 if lang_out == 'de' else 36
file_out = options.out_model
    # Setting for Model : Transformer
max_len = options.max_len
num_layers = options.num_layers
num_head = options.num_head
model_dim = options.model_dim
value_dim = options.value_dim
key_dim = options.key_dim
feed_dim = options.feed_dim
dropout = options.dropout
send_each = terminal_bool(options.send_each)
embed_share = terminal_bool(options.embed_share)
scale_embedding = terminal_bool(options.scale_embedding)
    # Setting for Training
label_smooth = options.label_smooth
lr = options.lr
total_cnt = options.total_step
mini_batch = options.mini_batch
total_batch = options.total_batch
warmup = options.warmup
    # Setting for Running
num_workers = options.num_workers
device = options.device
verbose = terminal_bool(options.verbose)

####   Call Tokenizer   ####
if lang_out == 'de':
    tokenizer = Tokenizer(BPE()).from_file( callpath(f"data/en_{lang_out}/{token_file}") )
else: ## 'fr'
    tokenizer = Tokenizer(WordPiece()).from_file( callpath(f"data/en_{lang_out}/{token_file}") )
tokenizer.enable_padding()

####   Call Data Path   ####
filenames = []
for i in range(num_txt):
    filenames.append(
            [callpath(f"data/{lang_in}_{lang_out}/train_{lan}_{i}.txt") for lan in (lang_in, lang_out)]
        )
print(f"Train with {len(filenames)} file(s).") if verbose else 0

####   HyperParameters(Most of them were already called by ArgParser)   ####
vocab_size = tokenizer.get_vocab_size()

model = Transformer(
    vocab_size=vocab_size, max_len=max_len+50, num_layers=num_layers,
    num_head=num_head, model_dim=model_dim, value_dim=value_dim,
    key_dim=key_dim, feed_dim=feed_dim, dropout=dropout, 
    send_each=send_each, embed_share=embed_share, scale_embedding=scale_embedding)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth, ignore_index=0)

trainer = Trainer(
    tokenizer=tokenizer, model=model, criterion=criterion, lr=lr,
    mini_batch=mini_batch, total_batch=total_batch, warmup=warmup, device=device)

trainer.verbose = verbose
trainer.train(train_cnt=total_cnt, src_lst=filenames, max_len=max_len, filename=callpath(f"result/{file_out}"), num_workers=num_workers)

b = trainer.test(src=[callpath(f"data/{lan_in}_{lan_out}/test_{lan}.txt") for lan in (lan_in, lan_out)], beam=True, max_len=50, num_workers=num_workers)
print(f"BLEU for {file_out} : {b*100 : .2f}")
# END