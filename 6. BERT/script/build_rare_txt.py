import sys
sys.path.insert(0,'..')

from tqdm import tqdm
from datasets import load_dataset

from torch.utils.data import DataLoader

from src.preprocess import clean_str_book, clean_str_wiki
from src.util import callpath

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit


"""
text from BOOKCORPUS
https://huggingface.co/datasets/bookcorpus
This page was run on datasets.2.10.0
"""

## It Doesn't Work Bcause of Version ISSUE.
dataset = load_dataset("bookcorpus")
loader = DataLoader(dataset['train']['text'])

# file id
fi = 1
end = False

f = open(callpath(f"data/rare/{fi}.txt"), 'w')
for i, data in enumerate(tqdm(loader, desc="Processing BookCorpus")):
    f.write(clean_str_book(data[0]))
    end = False
    ## Cut file approximate 1GB
    if not ((i+1) % 16000000):
        f.close()
        fi += 1
        f = open(callpath(f"data/rare/{fi}.txt"), 'w')
        end = True
f.close()

print(f"{fi} GB text were built")


"""
text from WIKIPEDIA
https://huggingface.co/datasets/wikipedia
"""


dataset = load_dataset("wikipedia", "20220301.en")
loader = DataLoader(dataset['train']['text'])

# Shift to next file
fj = fi + (1 - end)
l = 0

f = open(callpath(f"data/rare/{fi}.txt"), 'w')
for i, data in enumerate(tqdm(loader, desc="Processing WikiPedia")):
    out = clean_str_wiki(data[0])
    
    for o in out:
        f.write(o)
    
    f.write("\n")
    l += len(out)
    
    ## Cut file approximate 1GB
    if  l > 7000000:
        f.close()
        fj += 1
        f = open(callpath(f"data/rare/{fj}.txt"), 'w')
        l = 0
f.close()

print(f"{fj - fi} GB text were built")

########################
###### Tokenizer #######
########################

print("...building Tokenizer...")

filenames = [callpath(f"data/rare{i}.txt")
             for i in range(fj+1)]

tokenizer = Tokenizer(WordPiece(unk_token='<Unk>'))

tokenizer.pre_tokenizer = WhitespaceSplit()
trainer = WordPieceTrainer(vocab_size=30000, show_progress = True, special_tokens = ["<Pad>", "<Mask>", "<Cls>", "<Sep>", "<Unk>"])

tokenizer.train(filenames, trainer=trainer)

print(f"{tokenizer.get_vocab_size()} tokens are built!!!")

tokenizer.save(callpath("data/tokenizer.json"))