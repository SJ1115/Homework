import sys, os

sys.path.insert(0, "..")

from preprocess import Corpus
import pickle
from tqdm import tqdm


with open(os.path.join(sys.path[0],"corpus_save.pkl"), 'rb') as f:
    corpusBig, corpus30K = pickle.load(f,)

newCorpus = Corpus()

newCorpus.word_to_id = corpusBig.word_to_id
newCorpus.id_to_word = corpusBig.id_to_word
newCorpus.table = corpusBig.table

words_list = set(corpusBig.word_to_id.keys())

prefix = "onebilliondataset/news.en-"
suffix = "-of-00100"

new_prefix = '1_B/sub-'
new_suffix = '-of-100.pkl'

newCorpus.set_subsample()

for i in tqdm(range(0,100)):
    lines = []
    num = f"{i:05}"
    with open(os.path.join(sys.path[0], prefix + num + suffix), 'r') as f:
        while True:
            line = f.readline()
            #####
            if not line: break

            line = newCorpus.sample_corpus(line, words_list, False)
            if line:
                lines.append(line)
            
    pkl_file = new_prefix + num + new_suffix
    with open(os.path.join(sys.path[0],pkl_file), 'wb') as g:
        pickle.dump(lines, g)