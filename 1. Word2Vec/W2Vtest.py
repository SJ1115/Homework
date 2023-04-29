import sys, os

import numpy as np

from time import time
from datetime import timedelta

import pickle
from tqdm import tqdm

with open(os.path.join(sys.path[0],"analogy_test.pkl"), 'rb') as f:
    ###고정시켜버릴까?
    queries_ = pickle.load(f,)


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


class SemanticSyntacticTest:
    def __init__(self, word_to_id, id_to_word, word_matrix, queries=queries_):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.word_matrix = normalize(word_matrix)
        self.queries = queries
        self.queries_sem = self.queries[:8869]
        self.queries_syn = self.queries[8869:]
        self.result, self.result_sem, self.result_syn = [], [], []
        return
    
    def tovec(self, a):
        if a in self.word_to_id:
            return self.word_matrix[self.word_to_id[a]]
        else:
            return None

    def checkIn(self, word):
        if word in self.word_to_id:
            return self.word_to_id[word]
        return -1

    def analogy(self, a, b, c, d=None, n=5):
        print(f"ANALOGY test for {a}, {b}, {c}{(', and ' + d) if d is not None else None}")
        a, b, c = self.checkIn(a), self.checkIn(b),  self.checkIn(c)
        if d != None:
            d = self.checkIn(d)
        else:
            d = -2
        if not ((a+1) * (b+1) * (c+1) * (d+1)):
            raise ValueError("Word is not Valid")
        
        a_vec = self.word_matrix[a]
        b_vec = self.word_matrix[b]
        c_vec = self.word_matrix[c]

        query_vec = b_vec - a_vec + c_vec
        query_vec = normalize(query_vec)

        similarity = np.dot(self.word_matrix, query_vec)

        i = n
        best = []
        while i:
            loc = np.argmax(similarity)
            if (loc in (a, b, c)):
                similarity[loc] = 0
                continue
            print(f"{n-i+1} : score {similarity[loc]*100:.2f} : {self.id_to_word[loc]}")
            best.append(loc)
            similarity[loc] = 0
            i -= 1
        
        
        return (d in best)


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

    def test_0(self):
        sem_query = []; syn_query = []
        sem_sltn = [] ; syn_sltn = []
        sem_negl = [] ; syn_negl = []
        
        start = time()
        for query in self.queries_sem:
            a_vec = self.tovec(query['a'])
            b_vec = self.tovec(query['b'])
            c_vec = self.tovec(query['c'])
            d_sig = query['d'] in self.word_to_id
            if (a_vec is None) or (b_vec is None) or (c_vec is None) or not d_sig:
                continue
            sem_query.append(b_vec + c_vec - a_vec)
            sem_sltn.append(self.word_to_id[query['d']])
            sem_negl.append(np.array([self.word_to_id[query['a']], self.word_to_id[query['b']], self.word_to_id[query['c']]]))
        
        sem_query = normalize(np.array(sem_query))
        sem_answr = np.dot(sem_query, self.word_matrix.T)

        for i, neg_id in enumerate(sem_negl):
            sem_answr[i][neg_id] = 0

        sem_answr = np.argmax(sem_answr, axis=1)

        self.result_sem = np.equal(sem_answr, sem_sltn)
        print(f"{str(timedelta(seconds=int(time() - start)))} spent for Semantic Test")        
        ###

        start = time()
        for query in self.queries_syn:
            a_vec = self.tovec(query['a'])
            b_vec = self.tovec(query['b'])
            c_vec = self.tovec(query['c'])
            d_sig = query['d'] in self.word_to_id
            if (a_vec is None) or (b_vec is None) or (c_vec is None) or not d_sig:
                continue
            syn_query.append(b_vec + c_vec - a_vec)
            syn_sltn.append(self.word_to_id[query['d']])
            syn_negl.append(np.array([self.word_to_id[query['a']], self.word_to_id[query['b']], self.word_to_id[query['c']]]))

        syn_query = normalize(np.array(syn_query))
        syn_answr = np.dot(syn_query, self.word_matrix.T)

        for i, neg_id in enumerate(syn_negl):
            syn_answr[i][neg_id] = 0

        syn_answr = np.argmax(syn_answr, axis=1)

        self.result_syn = np.equal(syn_answr, syn_sltn)

        print(f"{str(timedelta(seconds=int(time() - start), ))} spent for Syntactic Test")
        
        self.result = np.append(self.result_sem, self.result_syn)
        return np.sum(self.result) / len(self.result)
        



