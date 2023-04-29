import sys, os
sys.path.insert(0,'..')
#from src.vocab import Vocab
from scipy.stats import spearmanr as corr
import numpy as np

from src.func import normalize
from src.vocab import Vocab
from time import time
from datetime import timedelta

import pickle
from tqdm import tqdm


## FILES

with open(os.path.join(os.path.dirname(__file__), '..', "dataset/test/analogy_test.pkl"), 'rb') as f:
    queries_ = pickle.load(f,)

test_WS353 =  {'w1':[], 'w2':[], 'sim':[]}
with open(os.path.join(os.path.dirname(__file__), '..', 'dataset/test/wordsim353_agreed.txt'), 'r') as f:
    #sys.path[0],"dataset/test/wordsim353_agreed.txt"), 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == '#':
            continue
        line = line.lower().split()
        test_WS353['w1'].append(line[1])
        test_WS353['w2'].append(line[2])
        test_WS353['sim'].append(float(line[3]))

test_RW =  {'w1':[], 'w2':[], 'sim':[]}
with open(os.path.join(os.path.dirname(__file__), '..', 'dataset/test/rw.txt'), 'r') as f:
    #sys.path[0],"dataset/test/wordsim353_agreed.txt"), 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        
        line = line.lower().split()
        test_RW['w1'].append(line[0])
        test_RW['w2'].append(line[1])
        test_RW['sim'].append(float(line[2]))

def wordSimTest(vocab, mode,info = True):
    """Parameter 'vocab' must be a class, with a function of 'vocab.find_vec'
    that inputs word(str) and outputs word vector.
    Parameter 'mod' must be "WS353" or "RW".
    """
    if mode.lower() == "ws353":
        test = test_WS353
    elif mode.lower() == "rw":
        test = test_RW
    else:
        raise ValueError("select PROPER test mode")

    ## TEST
    baseline = test['sim']
    a = []; b = []
    for w1, w2 in zip(test['w1'], test['w2']):
        a.append(vocab.find_vec(w1, info))
        b.append(vocab.find_vec(w2, info))
    a = np.array(a)
    b = np.array(b)
    vocab_result = np.einsum('ij,ij->i', a, b)
    return corr(baseline, vocab_result)
#####


class SemanticSyntacticTest:
    def __init__(self, data, is_param=False, queries=queries_):
        
        if is_param:
            self.vocab = Vocab()
            self.vocab.get_vocab(data)
        else:
            self.vocab = data
        
        self.queries = queries
        self.queries_sem = self.queries[:8869]
        self.queries_syn = self.queries[8869:]
        self.result, self.result_sem, self.result_syn = [], [], []
        return

    def analogy(self, a, b, c, d):

        a_vec = self.vocab.find_vec(a)
        b_vec = self.vocab.find_vec(b)
        c_vec = self.vocab.find_vec(c)
        d_vec = self.vocab.find_vec(d)
        if (a_vec is None) or (b_vec is None) or (c_vec is None) or (d_vec is None):
            return None
        
        query_vec = b_vec - a_vec + c_vec
        query_vec = normalize(query_vec)

        similarity = np.dot(self.vocab.word_matrix, query_vec)

        i = np.argmax(similarity)
        while(i in (self.vocab.word_to_id[a], self.word_to_id[b], self.word_to_id[c])):
            similarity[i] = 0
            i = np.argmax(similarity)
            
        if i == self.word_to_id[d]:
            return 1
        else:
            return 0
        
        similarity = np.dot(query_vec, d_vec)

        for i in range(self.word_matrix.shape[0]):
            if i in (self.word_to_id[a], self.word_to_id[b], self.word_to_id[c], self.word_to_id[d]):
                continue
            if similarity < np.dot(self.word_matrix[i], query_vec):
                return 0
        
        return 1


    def test(self):
        self.result_sem = []
        self.result_syn = []
        for query in tqdm(self.queries_sem):
            self.result_sem.append(self.analogy(query['a'], query['b'], query['c'], query['d']))
        for query in tqdm(self.queries_syn):
            self.result_syn.append(self.analogy(query['a'], query['b'], query['c'], query['d']))
        
        self.result_sem = [result for result in self.result_sem if result in (1, 0)]

        self.result_syn = [result for result in self.result_syn if result in (1, 0)]
        self.result = self.result_sem + self.result_syn
        return np.sum(self.result) / len(self.result)

    def if_in(self, word):
        if word in self.vocab.word_to_id:
            return 1
        else:
            return 0

    def test_0(self, ):
        sem_query = []; syn_query = []
        sem_sltn = [] ; syn_sltn = []
        sem_negl = [] ; syn_negl = []
        
        start = time()
        for query in self.queries_sem:
            if not np.product([self.if_in(q) for q in query.values()]):
                continue
            
            a_vec = self.vocab.find_vec(query['a'])
            b_vec = self.vocab.find_vec(query['b'])
            c_vec = self.vocab.find_vec(query['c'])

            sem_query.append(b_vec + c_vec - a_vec)
            sem_sltn.append(self.vocab.word_to_id[query['d']])
            sem_negl.append(np.array([self.vocab.word_to_id[query['a']], self.vocab.word_to_id[query['b']], self.vocab.word_to_id[query['c']]]))
        
        sem_query = normalize(np.array(sem_query))
        sem_answr = np.dot(sem_query, self.vocab.word_vecs.T)

        del sem_query ## for Memory eff.

        for i, neg_id in enumerate(sem_negl):
            sem_answr[i][neg_id] = 0

        sem_answr = np.argmax(sem_answr, axis=1)

        self.result_sem = np.equal(sem_answr, sem_sltn)
        print(f"{str(timedelta(seconds=int(time() - start)))} spent for Semantic Test")        
        ###

        start = time()
        for query in self.queries_syn:
            if not np.product([self.if_in(q) for q in query.values()]):
                continue
            a_vec = self.vocab.find_vec(query['a'])
            b_vec = self.vocab.find_vec(query['b'])
            c_vec = self.vocab.find_vec(query['c'])

            syn_query.append(b_vec + c_vec - a_vec)
            syn_sltn.append(self.vocab.word_to_id[query['d']])
            syn_negl.append(np.array([self.vocab.word_to_id[query['a']], self.vocab.word_to_id[query['b']], self.vocab.word_to_id[query['c']]]))

        syn_query = normalize(np.array(syn_query))
        syn_answr = np.dot(syn_query, self.vocab.word_vecs.T)

        del syn_query ## for Memory eff.

        for i, neg_id in enumerate(syn_negl):
            syn_answr[i][neg_id] = 0

        syn_answr = np.argmax(syn_answr, axis=1)

        self.result_syn = np.equal(syn_answr, syn_sltn)

        print(f"{str(timedelta(seconds=int(time() - start), ))} spent for Syntactic Test")
        
        self.result = np.append(self.result_sem, self.result_syn)
        return np.sum(self.result) / len(self.result)
        



