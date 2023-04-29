import sys, os

sys.path.insert(0,'..')

from tqdm import tqdm
import pandas as pd
import pickle

from preprocess import Corpus, huffman

now = "/hdd1/user6/Public/freshman/Word2Vec"
prefix = now + "/onebilliondataset/news.en-"
suffix = "-of-00100"

new_prefix = now + '/30_K/new-'
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

            t0, t1 = BigCorpus.add_corpus(line)

            lines.append(t0)
            tot_words += t1
            # t0 = 단어들, t1 = 단어수


print(f"We have total {tot_words} words, with {BigCorpus.vocab_size}")

freq = pd.Series(BigCorpus.table).sort_values(ascending=False)
threshold = freq[30000]

freq_30K = freq[freq > threshold-1]
words_30K = list(freq_30K.keys())

with open(now + '/word_list_30K.pkl', 'wb') as f:
    pickle.dump(freq_30K, f, )

print("Second, run again with word_list restricked 30K")


Corpus_30K = Corpus()
tot_words_30K = 0 ## len of total corpus

for i in tqdm(range(0,100)):
    lines = []
    num = f"{i:05}"
    with open(prefix + num + suffix, 'r') as f:
        while True:
            line = f.readline()
            #####
            line , line_len= Corpus_30K.add_corpus(line, allow = words_30K)
            if not line: break

            lines.append(line)
            tot_words_30K += line_len
    pkl_file = new_prefix + num + new_suffix
    with open(pkl_file, 'wb') as g:
        pickle.dump(lines, g)

print(f"Now, we got total {tot_words_30K} words, with vocab {Corpus_30K.vocab_size}")

param = {}
param["word_to_id"] = BigCorpus.word_to_id
param["id_to_word"] = BigCorpus.id_to_word

with open(now + "/corpus_save.pkl", 'wb') as f:
    pickle.dump([BigCorpus, Corpus_30K], f)

print("for Later, build huffman tree with Corpus")

## run huffman with full word
HuffBig = huffman(freq=BigCorpus.table)
HuffBig.build_codebook()

with open(now + '/huffman_full.pkl', 'wb') as f:
    pickle.dump(HuffBig, f)

print("Huffman Tree of Full Vocab is Done!")
## run huffman with 30K word
Huff_30K = huffman(freq=Corpus_30K.table)
Huff_30K.build_codebook()

with open(now + '/huffman_30K.pkl', 'wb') as f:
    pickle.dump(Huff_30K, f)

print("HUffman Tree for 30K words is Done!\nAll Preprocessing is Done!!")