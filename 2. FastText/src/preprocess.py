import numpy as np
import pandas as pd
import copy
import re
import operator
from tqdm import tqdm
import csv

"""
1. Corpus의 빈도 통계 구하기/ 입력하기
    를 통해 Huffman Tree 만들기
"""
def split(a):

    k, m = divmod(len(a), 2)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(2))

def re_split(a):
    
    if len(a) <= 1:
        return a[0]
    return [re_split(i) for i in split(a)]

class huffman:
 
    """코퍼스는 쭉 편 상태여야 함(겹_리스트 형태인 경우, 늘려서 넣을 것)"""
    def __init__(self, corpus=None, freq = None, order=True, verbose = False):
        if (corpus is None) and (freq is None):
            ValueError("Put Corpus or Freq-Table In")
            return
        if (freq is None) and (corpus is not None):
            self.freq = dict(pd.Series(corpus).value_counts())
        elif (freq is not None) and (corpus is None):
            self.freq = freq
        else:
            ValueError("Put Either Corpus or Freq-Table, Not Both")
        self.codebook = {} ## Huffman code를 보관
        self.nodebook = {} ## huffman code를 따라 지나가게 될 노드 위치를 보관
        self.tree = None
        self.node_num = None
        self.params, self.grads = [], []

        Val = list(self.freq.values())
        Key = list(self.freq.keys())
        # Sort it at first
        Val, Key = list(map(list, zip(*sorted(zip(Val, Key), key=lambda x: x[0]))))

        if order:
            Key.sort()
            self.tree = re_split(Key)
        else:
            if verbose:
                timeiter = tqdm(range(len(self.freq) -1))
            else:
                timeiter = range(len(self.freq) -1)
            for i in timeiter:
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
    
    def preprocess(self, doc, allow = None, only_freq = False):
        """Preprocessing fog Huggingface Wiki Data.
        """
        dic = {
            "&": ' and ',   "\n":'',        "-":' ',
            ",": ' ,',      "'s": " 's",
            "n't" : " not", "'m": " am",    "ms.":"ms ",
            "mr.":"mr ",    "dr.": "dr ",   "inc.": "inc ",
            "~":" to ",     "£":'',         "$":'', "@":'',
            "p.m.":"pm ",   "pm.":"pm ",    "p. m.":"pm ", 
            "a.m.":"am ",   "am.":"am ",    "a. m.":"am ",
        }
        
        lines = re.split('\.\n|\. ', doc)

        lines_out = []
        nWords_out = 0
        for line in lines:
            line = line.lower() + ' EOS'
            #### Preprocessing
            for i, j in dic.items():
                line = line.replace(i,j)

            ## unify number to 'N' token
            line = re.sub('((-)?\d{1,3}(,\d{3})*(\.\d+)?)', 'N', line)
            line = re.sub('N{1,}', ' N ', line)
            line = re.sub(r'[^a-z A-Z.]', ' ', line)
            line = re.sub(' {2,}', ' ', line)
            ###
            words = line.split() 
            if allow != None:
                words = [w for w in words if w in allow]
                
            lines_out.append(words)
            nWords_out += len(words)
        return lines_out, nWords_out

    def add_corpus(self, doc, allow=None, preprocessed = False):
        
        if preprocessed:
            lines = doc
            lines_len = len(doc)
        else:
            lines, lines_len = self.preprocess(doc = doc, allow=allow)

        for line in lines:
            for word in line:
                if word not in self.word_to_id:
                    new_id = self.vocab_size

                    self.word_to_id[word] = new_id
                    self.id_to_word[new_id] = word
                    self.table[word] = 0
                    self.vocab_size += 1

                else:
                    self.table[word] += 1
        
        return [[self.word_to_id[word] for word in line] for line in lines], lines_len

    def __pop_word(self, id):
        """It only pop word_dictionary, so the id list would be gapped.
        You should manage to it if you call this function."""
        self.table.pop(self.id_to_word[id], None)
        self.word_to_id.pop(self.id_to_word[id], None)
        self.id_to_word.pop(id, None)

    def reduce_corpus(self, corpus, min_count = 5):
        transit = {}
        remove = set()
        i = 0
        #timeiter = tqdm(total=self.vocab_size, desc="removing sparse words")
        while(i < self.vocab_size):
            if self.table[self.id_to_word[i]] < min_count:
                while(self.table[self.id_to_word[self.vocab_size-1]] < min_count):
                    self.__pop_word(self.vocab_size-1)
                    self.vocab_size -= 1
                    remove.add(self.vocab_size)
                    
                if i >= self.vocab_size:
                    break
                self.__pop_word(i)
                
                self.id_to_word[i] = self.id_to_word[self.vocab_size-1]
                self.word_to_id[self.id_to_word[self.vocab_size-1]] = i
                
                self.id_to_word.pop(self.vocab_size-1, None)
                self.vocab_size -= 1
                
                
                transit[self.vocab_size] = i
                remove.add(self.vocab_size)
            
            i += 1
        
        for i, line in enumerate(tqdm(corpus, desc="cropping the original sentences")):
            corpus[i] = [transit[word] if word in transit else word for word in line if word not in remove]
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
            !!!! Before call this function, you must call 'set_subsample' once.
        """

        if preprocessed:
            words = line
            words_len = len(line)
        else:
            words, words_len = self.preprocess(line=line, allow=allow)

        ## Random Number in [0, 1]
        threshold = np.random.random(len(words))
        if preprocessed:
            return [word for word, t in zip(words, threshold) if self.threshold[word] < t]
        else:
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
            if s <  2 * window_size + 1:
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

def create_contexts_target(corpus, window_size=1, vervose = True):
    '''맥락과 타깃 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    '''
    target = []
    contexts = []

    if vervose:
        timeiter = tqdm(corpus, desc="Building Train set")
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


############################################
### From Here is for Text Classification ###
############################################
import sys, os

def data_to_param(data):
    if data == 'ag':
        data_src = "dataset/classification/ag_news_csv"
        data_fld = {'class': [], 'title': [], 'text': []}
    elif data == 'amazon_f':
        data_src = "dataset/classification/amazon_review_full_csv"
        data_fld = {'class': [], 'title': [], 'text': []}
    elif data == 'amazon_p':
        data_src = "dataset/classification/amazon_review_polarity_csv"
        data_fld = {'class': [], 'title': [], 'text': []}
    elif data == 'dbpedia':
        data_src = "dataset/classification/dbpedia_csv"
        data_fld = {'class': [], 'title': [], 'text': []}
    elif data == 'sogou':
        data_src = "dataset/classification/sogou_news_csv"
        data_fld = {'class': [], 'title': [], 'text': []}
    elif data == 'yahoo':
        data_src = "dataset/classification/yahoo_answers_csv"
        data_fld = {'class': [], 'title': [], 'question': [], 'answer': []}
    elif data == 'yelp_f':
        data_src = "dataset/classification/yelp_review_full_csv"
        data_fld = {'class': [], 'text': []}
    elif data == 'yelp_p':
        data_src = "dataset/classification/yelp_review_polarity_csv"
        data_fld = {'class': [], 'text': []}
    else:
        ValueError("data name is not VALID: keyword [data] should be one of [\"ag\"], [\"amazon_p\"], [\"amazon_f\"], [\"dbpedia\"], [\"sogou\"], [\"yahoo\"], [\"yelp_p\"] or [\"yelp_f\"]")
    
    return data_src, data_fld

def clean_str(string):
  """
  Tokenization/string cleaning for all data except for Chinese.
  Since there exist accent / Cyrill letters, I skipped the Removal of non-English.
  """
#  string = re.sub(r"[;\-_:\*@#%\^&$<>\[\]|\+\=]", " ", string)

  string = re.sub(r"\.{2,}", " ", string)
  string = re.sub(r"[^\w(),\.\\!?\']", " ", string)
  string = re.sub(r"[0-9]", "", string) 
  string = re.sub(r"\'s", " s", string) 
  string = re.sub(r"\'ve", " have", string) 
  string = re.sub(r"n\'t", " not", string) 
  string = re.sub(r"\'re", " re", string) 
  string = re.sub(r"\'d", " d", string) 
  string = re.sub(r"\'ll", " will", string)
  string = re.sub(r"\\n", " ", string)
  string = re.sub(r"\\", " ", string)
  
  string = re.sub(r",", " , ", string)
  string = re.sub(r"/", " / ", string)
  string = re.sub(r"_", " ", string)
  string = re.sub(r"\'", " ", string)
  string = re.sub(r"\.", " . ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

def clean_str_sogou(string):
  """
  Tokenization/string cleaning for Chinese.
  """
  string = ' ' + string
  string = re.sub(r"[^A-Za-z0-9(),.!?]", " ", string)     
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"\.", " . ", string)
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r" \d+", " ", string)
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

def text_to_words(text, data):
    if data == 'sogou':
        clean_text = clean_str_sogou(text)
    else:
        clean_text = clean_str(text)
    words = clean_text.split(' ')
    return words

def get_vocab(dir_lst, fld, data, threshold):
    word_to_idx = {}
    freq = {}
    ## 0 for Unknown
    idx = 1

    for directory in dir_lst:
        f = open(os.path.join(os.path.dirname(__file__), '..', directory) , "r",)
        reader = csv.DictReader(f, fieldnames=fld)
        for line in reader:
            for key, value in line.items():
                if key != 'class':
                    words = text_to_words(value, data)
                    for word in words:
                        if not word in word_to_idx:
                            word_to_idx[word] = idx
                            freq[word] = 1
                            idx += 1
                        else:
                            freq[word] += 1
        f.close()
    for word in freq:
        if freq[word] < threshold:
            word_to_idx.pop(word)
    if threshold:
        for word, id in zip(word_to_idx.keys(), range(len(word_to_idx))):
            word_to_idx[word] = id + 1
    return word_to_idx

def preprocess_TC(data, threshold=3):
    data = data.lower()

    src, fld = data_to_param(data)
    if data == 'sogou':
        csv.field_size_limit(500000)
    
    dir_lst = [src + fname for fname in ['/train.csv', '/test.csv']]

    word_to_idx = get_vocab([dir_lst[0]], fld, data, threshold=threshold)

    data_out = {}
    data_out['w2i'] = word_to_idx

    print(f"Vocab size is {len(word_to_idx)}")

    for mode, dir_ in zip(['train', 'test'], dir_lst):
        text = []
        label = []
        
        with open(os.path.join(os.path.dirname(__file__), '..', dir_), 'r') as f:
            reader = csv.DictReader(f, fieldnames=fld)
            
            for line in reader:
                sent = []
                for key, value in line.items():
                    if key == 'class':
                        label.append(int(value))
                    else:
                        words = text_to_words(value, data)
                        sent += [word_to_idx[word] if word in word_to_idx else 0 for word in words ]

                text.append(np.array(sent))

        data_out[mode] = np.array(text, dtype=object)
        data_out[mode + '_label'] = np.array(label) - 1 ## It starts at 1
  
    return data_out

def corpus_bigram(data, bigram=False):
    if bigram:
        corpus = np.append(data['train'], data['test'])
        w2i = {}
        ind = 0
        for sent in corpus:
            for i in range(len(sent)-1):
                tempa, tempb = sent[i], sent[i+1]
                biId = (tempa, tempb) if tempa < tempb else (tempb, tempa)
                if biId not in w2i:
                    w2i[(biId)] = ind
                    ind += 1
        
        train = []
        for sentId in range(len(data['train_label'])):
            sent = data['train'][sentId]
            new_sent = []
            for i in range(len(sent)-1):
                tempa, tempb = sent[i], sent[i+1]
                biId = (tempa, tempb) if tempa < tempb else (tempb, tempa)
                new_sent.append(w2i[biId])
            train.append(np.array(new_sent))

        test = []
        for sentId in range(len(data['test_label'])):
            sent = data['test'][sentId]
            new_sent = []
            for i in range(len(sent)-1):
                tempa, tempb = sent[i], sent[i+1]
                biId = (tempa, tempb) if tempa < tempb else (tempb, tempa)
                new_sent.append(w2i[biId])
            test.append(np.array(new_sent))

        train = np.array(train, dtype=object)
        test = np.array(test, dtype=object)

        ## Remove 0-len sentence
        ##### It occurs only when original sentence is len 1
        train_label = data['train_label'][np.vectorize(len)(train)>0]
        test_label = data['test_label'][np.vectorize(len)(test)>0]
        train = train[np.vectorize(len)(train)>0]
        test = test[np.vectorize(len)(test)>0]
    else:
        train = data['train']
        test = data['test']
        w2i = data['w2i']
        train_label = data['train_label']
        test_label = data['test_label']

    return w2i, train, train_label, test, test_label