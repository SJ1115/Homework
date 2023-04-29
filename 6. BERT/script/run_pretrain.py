import sys, os
sys.path.insert(0, "..")

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
import argparse

from src.model import BERT_LM
from src.config import Config_mini as Config
from src.trainer import Trainer
from src.util import callpath, terminal_bool

########## SETTING ##########
parser = argparse.ArgumentParser(description="----BERT-Pre-Train----")

parser.add_argument("--out_model", default="sample.pt", type=str, help="filename for output model.pt")

parser.add_argument("--device", default='cuda:0', type=str, choices = ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help="device where training is executed")
parser.add_argument("--use_tqdm", default='t', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to show progress bar")
parser.add_argument("--use_board", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to use TensorBoard")
parser.add_argument("--is_pre_norm", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to lay a Norm on : default) False")

parsed_opt = parser.parse_args()

Config.filename = callpath(f"result/" + parsed_opt.out_model)
Config.device = parsed_opt.device
Config.use_tqdm = terminal_bool(parsed_opt.use_tqdm)
Config.use_board = terminal_bool(parsed_opt.use_board)
Config.src_dir = callpath("data/done/")
Config.is_pre_norm = terminal_bool(parsed_opt.is_pre_norm)

########## PRE_TRAIN ##########

model = BERT_LM(Config)

trainer = Trainer(model, Config)

trainer.pre_train(Config)
