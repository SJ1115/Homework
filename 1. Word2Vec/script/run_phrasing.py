import sys, os
import pickle
from copy import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict as Dict

### Load Original Corpus
with open(os.path.join(sys.path[0],"corpus_save.pkl"), 'rb') as f:
    corpusBig, corpus30K = pickle.load(f,)

V = corpusBig.vocab_size
min_count = 5
threshold = 100
### build freq table
id_freq = copy(corpusBig.id_to_word)
ph_freq = {}
for i in id_freq:
    id_freq[i] = corpusBig.table[id_freq[i]]
    ph_freq[i] = Dict(int)



prefix = "1_B/phrases/sub-"
suffix = "-of-100.pkl"


new_phrases = 0
timeiter = tqdm(total= V)

## collecting all 2-gram
total_words = 0
for i in range(100):
    ##openfile
    with open(os.path.join(sys.path[0], prefix + f"{i:05}" + suffix), 'rb') as f:
        lines = pickle.load(f, )
    for sentence in lines:
        for w_ind in range(len(sentence)-2):
            ## get freq of next word of w
            ph_freq[sentence[w_ind]][sentence[w_ind + 1]] += 1
        total_words += len(sentence)

w = 0
while w < V:
    timeiter.update(1)
    timeiter.set_description(f"for {corpusBig.id_to_word[w]}")
    if not corpusBig.table[corpusBig.id_to_word[w]]:
        continue
    freq_ph = copy(id_freq)




    for w_ in freq_ph:
        if freq_ph[w_] < 1: # not phrasing a word with less than min_count
            continue

        if not id_freq[w_]:
            continue
        ## count(ab) / count(a) / count(b) * words_size(783M) > threshold(100)
        score = freq_ph[w_] * total_words / corpusBig.table[corpusBig.id_to_word[w]] / corpusBig.table[corpusBig.id_to_word[w_]]

        if score > threshold:
            ## add phrases to corpus
            new_phrase = corpusBig.id_to_word[w] + ' ' + corpusBig.id_to_word[w_]
            corpusBig.id_to_word[V] = new_phrase
            corpusBig.word_to_id[new_phrase] = V
            corpusBig.table[new_phrase] = freq_ph[w_]
            corpusBig.table[corpusBig.id_to_word[w]] -= freq_ph[w_]
            corpusBig.table[corpusBig.id_to_word[w_]] -= freq_ph[w_]
            total_words -= freq_ph[w_]
            timeiter.set_description(f"for {new_phrase}")
            
            ## Update corpus lines
            for i in range(100):
                ##openfile
                with open(os.path.join(sys.path[0], prefix + f"{i:05}" + suffix), 'rb') as f:
                    lines = pickle.load(f, )
                for sentence in lines:
                    for w_ind in range(len(sentence)-2):
                        ## get freq of next word of w
                        if sentence[w_ind] == w and sentence[w_ind + 1] == w_:
                            sentence.pop(w_ind)
                            sentence[w_ind] = V

                #with open(os.path.join(sys.path[0], prefix + f"{i:05}" + suffix), 'wb') as f:
                #    pickle.dump(lines, f, )
            
            V += 1
            corpusBig.vocab_size += 1
            new_phrases += 1
    w += 1
    #print("")

print(f"{new_phrases} phrases are found!!")
with open(os.path.join(sys.path[0],"corpus_phrase.pkl"), 'rb') as f:
    corpusBig = pickle.load(f,)