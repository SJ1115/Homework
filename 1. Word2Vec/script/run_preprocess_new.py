import sys, os

sys.path.insert(0,'..')

from tqdm import tqdm
import pandas as pd
import pickle

from preprocess import Corpus, huffman

now = "/hdd1/user6/Public/freshman/Word2Vec"
prefix = now + "/onebilliondataset/news.en-"
suffix = "-of-00100"

new_prefix = now + '/1_B/new-'
new_suffix = '-of-100.pkl'

BigCorpus = Corpus()

tot_words = 0

print("First, run for words statistic")
for i in tqdm(range(100)):
    filename = prefix + f'{i:05}' + suffix

    lines = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()

            if not line: break

            line, l = BigCorpus.add_corpus(line)

            lines.append(line)
            tot_words += l
            # t0 = 단어들, t1 = 단어수

    newfile = new_prefix + f'{i:05}' + new_suffix
    with open(newfile, 'wb') as f:
        pickle.dump(lines, f)

print(f"We have total {tot_words} words, with {BigCorpus.vocab_size}")

BigCorpus.reduce_corpus()

print(f"But we use only {BigCorpus.vocab_size} words")

for i in tqdm(range(100)):
    newfile = new_prefix + f'{i:05}' + new_suffix

    with open(newfile, 'rb') as f:
        lines = pickle.load(f)

    lines = BigCorpus.filter_corpus(lines)
        
    with open(newfile, 'wb') as f:
        pickle.dump(lines, f)


BigCorpus.get_freq_list()

param = {}
param["word_to_id"] = BigCorpus.word_to_id
param["id_to_word"] = BigCorpus.id_to_word

with open(now + "/corpus_new.pkl", 'wb') as f:
    pickle.dump(BigCorpus, f)

print("for Later, build huffman tree with Corpus")

## run huffman with full word
HuffBig = huffman(freq=BigCorpus.table)
HuffBig.build_codebook()

with open(now + '/huffman_full.pkl', 'wb') as f:
    pickle.dump(HuffBig, f)

print("Huffman Tree of Full Vocab is Done!\nAll Preprocessing is Done!!")

