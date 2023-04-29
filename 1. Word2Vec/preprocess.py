import numpy as np
import pandas as pd
import copy
import re
import operator
from tqdm import tqdm

"""
1. Corpus의 빈도 통계 구하기/ 입력하기
    를 통해 Huffman Tree 만들기
"""


class huffman:
 
    """코퍼스는 쭉 편 상태여야 함(겹_리스트 형태인 경우, 늘려서 넣을 것)"""
    def __init__(self, corpus=None, freq = None):
        if (corpus is None) and (freq is None):
            print("ERROR")
            return
        if (freq is None) and (corpus is not None):
            self.freq = dict(pd.Series(corpus).value_counts())
        elif (freq is not None) and (corpus is None):
            self.freq = freq
        self.codebook = {} ## Huffman code를 보관
        self.nodebook = {} ## huffman code를 따라 지나가게 될 노드 위치를 보관
        self.tree = None
        self.node_num = None
        self.params, self.grads = [], []

        Val = list(self.freq.values())
        Key = list(self.freq.keys())
        # Sort it at first
        Val, Key = list(map(list, zip(*sorted(zip(Val, Key), key=lambda x: x[0]))))

        for i in tqdm(range(len(self.freq) -1)):
            #merge 2 node(key) to one,
            #sum 2 weight(val) as root's weight
            Key[0] = [Key.pop(0), Key[0]]
            Val[0] = Val.pop(0) + Val[0]

            #Sort lists(Key, Val) again only by pushing 0-th element to adequate spot
            j = 0
            while(Val[0] > Val[j]):
                j += 1
                if j == len(Val)-1: break
            
            Key.insert(j-1, Key.pop(0))
            Val.insert(j-1, Val.pop(0))

        self.tree = Key[0]
        self.timer = tqdm
        return

    """  """
    def build_codebook(self):
        self.codebook = {}
        self.nodebook = {} 

        self.node_num = 0
        self.build_code(self.tree, self.codebook, self.nodebook, [], [])

        return self.codebook
    

    def build_code(self, tree, codebook, nodebook, current_code, current_node):
        current_node = current_node + [self.node_num]
        self.node_num += 1
        

        if isinstance(tree[0], str) or isinstance(tree[0], int):
            codebook[tree[0]] = current_code + [0]
            nodebook[tree[0]] = current_node
        else:
            self.build_code(tree[0], codebook, nodebook, current_code + [0], current_node)
        
        if isinstance(tree[1], str) or isinstance(tree[1], int):
            codebook[tree[1]] = current_code + [1]
            nodebook[tree[1]] = current_node
        else:
            self.build_code(tree[1], codebook, nodebook, current_code + [1], current_node)

####
"""
2. 텍스트 전처리
"""

class Corpus:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.table = {}
        self.corpus = []
        self.vocab_size = 0
        self.threshold = {}
        self.freq_list = []
        self.transit = {}
        self.remove = set()
        
    def preprocess(self, line, allow = None, only_freq = False):
        dic = {
            "&":'and', "\n":'', "-":' ', "n 't":" not", "'m":" am",
            "ms.":"ms ", "mr.":"mr ", "dr.":"dr ", "inc.": "inc ",
            "~":" to ", "£":'', "$":'', "@":'', "p.m.":"pm ",
            "pm.": "pm ", "p. m.":"pm ", "a.m.":"am ",
            "am.": "am ", "a. m.":"am "
        }
        line = line.lower() + " EOS"
        #### Preprocessing
        for i, j in dic.items():
            line = line.replace(i,j)

        line = re.sub('((-)?\d{1,3}(,\d{3})*(\.\d+)?)', 'N', line)
        line = re.sub('N{1,}', 'N ', line)
        line = re.sub(' {2,}', ' ', line)
        ###
        line = re.sub(r'[^a-z A-Z.]', ' ', line)
        line = re.sub(' {2,}', ' ', line)
        words = line.split()
        if allow != None:
            words = [w for w in words if w in allow]
            
        return words, len(words)

    def add_corpus(self, line, allow=None, preprocessed = False):
        
        if preprocessed:
            words = line
            words_len = len(line)
        else:
            words, words_len = self.preprocess(line=line, allow=allow)

        
        for word in words:
            if word not in self.word_to_id:
                new_id = self.vocab_size

                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
                self.table[word] = 0
                self.vocab_size += 1

            else:
                self.table[word] += 1
        
        return [self.word_to_id[word] for word in words], words_len

    def __pop_word(self, id = -1):
        """It only pop word_dictionary, so the id list would be gapped.
        You should manage to it if you call this function."""
        if id == -1:
            ## default of id = self.vocab_size-1(Last Word)
            id += self.vocab_size
        self.table.pop(self.id_to_word[id], None)
        self.word_to_id.pop(self.id_to_word[id], None)
        self.id_to_word.pop(id, None)
                    

    def reduce_corpus(self, min_count = 5):
        self.transit = {}
        self.remove = set()
        i = 0
        timeiter = tqdm(total = self.vocab_size, desc='reducing sparse words..')
        while(i < self.vocab_size):
            if self.table[self.id_to_word[i]] < min_count:
                while(self.table[self.id_to_word[self.vocab_size-1]] < min_count):
                    self.__pop_word()
                    self.vocab_size -= 1
                    self.remove.add(self.vocab_size)
                    timeiter.update(1)
                
                if i>= self.vocab_size:
                    break
                self.__pop_word(i)

                self.id_to_word[i] = self.id_to_word[self.vocab_size-1]
                self.word_to_id[self.id_to_word[self.vocab_size-1]] = i

                self.id_to_word.pop(self.vocab_size-1, None)
                self.vocab_size -= 1
                timeiter.update(1)

                self.transit[self.vocab_size] = i
                self.remove.add(self.vocab_size)

            i += 1
            timeiter.update(1)

    def filter_corpus(self, corpus):
        for i, line in enumerate(corpus):
            corpus[i] = [self.transit[wid] if wid in self.transit else wid for wid in line if wid not in self.remove]
        return corpus

    def get_freq_list(self, power = 3/4):
        """it makes freq list that supplement of Negative Sampling.
        power(default = 3/4) is a exponent for decreasing.
        """
        word_id = 0
        self.freq_list = []
        while word_id in self.id_to_word:
            if word_id == self.word_to_id['EOS']:
                word_id += 1
                continue
            times = int(pow(self.table[self.id_to_word[word_id]], 3/4)+1)
            self.freq_list += [word_id] * times
            word_id += 1
        self.freq_list = np.array(self.freq_list)
    
    def set_subsample(self, t = 1e-5):
        """before subsampling, it builds threshold table with own frequency table.
        you have to input threshold t(default = 0.00001)."""
        self.threshold = {}
        
        tot = len(self.table)
        for w in self.table:
            tot += self.table[w]
        
        for w in self.table:
            if w == self.word_to_id["EOS"]:
                ## Do not erase EOS
                self.threshold[self.word_to_id[w]] = 0
            else:
                self.threshold[self.word_to_id[w]] = max(0, 1 - np.sqrt(t / (self.table[w]+1) * tot))
        return
    
    def sample_corpus(self, line, allow=None ,preprocessed = False):
        """SubSampling the Corpus.
            a word with smaller freq than teh threshold 't'(default = 1e-5) is un-deletable
            a word with bigger freq than ~~ might be gonna deleted.
            !!!! Before call this function, you must call 'set_subsample' first.
        """

        if preprocessed:
            words = line
            words_len = len(line)
        else:
            words, words_len = self.preprocess(line=line, allow=allow)

        ## Random Number in [0, 1]
        threshold = np.random.random(len(words))
        
        return [self.word_to_id[word] for word, t in zip(words, threshold) if self.threshold[word] < t]

    def create_contexts_target(self, corpus, window_size=1, subsampling = True, vervose = True):
        '''맥락과 타깃 생성

        :param corpus: 말뭉치(단어 ID 목록)
        :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
        :return:
        '''
        target = []
        contexts = []

        if subsampling:
            ## for efficiency, just use 200
            threshold = np.random.random(200)
            j = 0

        if vervose:
            timeiter = tqdm(corpus, desc="Building Training Set")
        else:
            timeiter = corpus
        for sentence in timeiter:
            s = len(sentence)
            if s <  window_size + 1:
                continue

            for i, word in enumerate(sentence):
                cs = []
                for t in range(max(0, i - window_size), min(s, i + window_size + 1)):
                    if t == i:
                        continue

                    if subsampling:
                        j = (j + 1) % 200
                        if self.threshold[sentence[t]] > threshold[j]:
                            continue
                    cs.append(sentence[t])

                if len(cs) != 0:
                    contexts.append(np.array(cs))
                    target.append(word)

        return contexts, target

def mix(a, b, vervose = True):
    """it will mix two list.
    two lists must have same length."""
    if len(a) != len(b):
        raise Exception("Two lists must have same Length!")
    
    k = int(np.sqrt(len(a)))+1

    if vervose:
        timeiter = tqdm(np.random.randint(0, len(a), (k, 2)) , desc='Mixxing Train File...')
    else:
        timeiter = np.random.randint(0, len(a), (k, 2))
    for i, j in timeiter:
        a[i], a[j] = a[j], a[i]
        b[i], b[j] = b[j], b[i]
    
    return a, b