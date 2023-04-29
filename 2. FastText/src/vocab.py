from collections import defaultdict as Dict
from src.preprocess import Corpus
from src.func import normalize
from tqdm import tqdm
import numpy as np

BOW = "<"
EOW = ">"

class Word:
    def __init__(self, word, id = None):
        self.word = word
        self.count = 1
        self.id = id
        self.type = False # F for word, T for label
        self.subwords = [] #????
    
    def get_subword(self, n_gram):
      if self.word in ('EOS', 'N'):
        self.subwords.append(self.word)
        ## Do not split token
        return
      else:
        word = BOW + self.word + EOW
        length = len(word) + 1
        if length <= np.min(n_gram):
          self.subwords.append(self.word)
        for n in n_gram:
          if length > n:
            for i in range(length-n):
              subword = ''.join(word[i:i+n])
              self.subwords.append(subword)
        
      return

class Vocab(Corpus):
    def __init__(self, n = [i for i in range(3,7)], **kwargs):
        Corpus.__init__(self, **kwargs)

        self.words = {}
        
        self.subword_to_subid = {}
        self.subid_to_subword = {}
        self.id_to_subid = Dict(set)
        
        self.subword_table = Dict(int)

        self.n_gram = n
        self.sub_vocab_size = 0

        self.subword_vecs = None
        self.word_vecs = None
     
    def get_from_corpus(self, corpus=None):
      if corpus:
        self.word_to_id = corpus.word_to_id
        self.id_to_word = corpus.id_to_word
        self.table = corpus.table
        self.vocab_size = corpus.vocab_size

      for word, id in tqdm(self.word_to_id.items(), desc="collecting subwords"):
        tempwrd = Word(word, id)
        tempwrd.count = self.table[word]
        tempwrd.get_subword(self.n_gram)

        self.add_subword(tempwrd)

        self.words[id] = tempwrd
      
      for id in self.id_to_subid:
        ## it has the same result with np.array(list(set)), but is faster.
        self.id_to_subid[id] = np.fromiter(self.id_to_subid[id], int, len(self.id_to_subid[id]))


    def add_subword(self, word):
      """word is class 'Word'"""
      for subword in word.subwords:
        
        self.subword_table[subword] += word.count
        if subword not in self.subword_to_subid:
          subid = self.sub_vocab_size
          self.subword_to_subid[subword] = self.sub_vocab_size
          self.subid_to_subword[self.sub_vocab_size] = subword
          self.sub_vocab_size += 1
        else:
          subid = self.subword_to_subid[subword]
        self.id_to_subid[word.id].add(subid)
          

    def wid_to_swid(self, sentence):
      result = []
      for wid in sentence:
        result += self.words[wid].subwords
      result = [self.subword_to_subid[r] for r in result]
      return result

  ### Build Word Matrix
    def build_wordvec(self, subword_vec, word_vec, a=True, b=True, c=True):
      self.subword_vecs = subword_vec
      if a:
        self.subword_vecs = normalize(self.subword_vecs)
      #self.subword_vecs = subword_vec

      self.word_vecs = np.zeros_like(word_vec)
      for id, subid in tqdm(self.id_to_subid.items(), desc="gathering subword_vectors", leave=False):
        for si in subid:
          self.word_vecs[id] += self.subword_vecs[si]
      if b:
        self.word_vecs = normalize(self.word_vecs)
      
      self.word_vecs += normalize(word_vec)
      if c:
        self.word_vecs = normalize(self.word_vecs)
      return 
    
    def new_wordvec(self, subword_vec, word_vec, a=True, b=True, c=True):
      self.subword_vecs = subword_vec
      if a:
        self.subword_vecs = normalize(self.subword_vecs)
      #self.subword_vecs = subword_vec
      word_vec = normalize(word_vec)

      self.word_vecs = np.zeros_like(word_vec)
      for id, subid in tqdm(self.id_to_subid.items(), desc="gathering subword_vectors", leave=False):
        for si in subid:
          self.word_vecs[id] += self.subword_vecs[si]
        if b:
          self.word_vecs[id] += len(subid)*word_vec[id]
      
      if not b:
        self.word_vecs += word_vec
      if c:
        self.word_vecs = normalize(self.word_vecs)
      return 

    def find_vec(self, word, info = True):
      if not isinstance(self.word_vecs, np.ndarray):
        raise Exception("Built word matrix First")

      if word in self.word_to_id:
        return self.word_vecs[self.word_to_id[word]]
      
      else:
        ### OOV case
        if info:
          tempwrd = Word(word)
          tempwrd.get_subword(self.n_gram)
          vec = np.zeros(self.word_vecs.shape[1])
          for sw in tempwrd.subwords:
            if sw in self.subword_to_subid:
              vec += self.subword_vecs[self.subword_to_subid[sw]]
        else:
          vec = np.zeros_like(self.word_vecs[0])        
        return normalize(vec)

  ### When we read file, build vocab directly(It's for WS TEST)
    def get_vocab(self, params, a=True, b=True, c=True):
      self.word_to_id = params['word_to_id']
      self.id_to_word = params['id_to_word']
      self.vocab_size = len(self.id_to_word)

      if 'id_to_subid' in params:
        self.id_to_subid = params['id_to_subid']
        self.subid_to_subword = params['subid_to_subword']
        self.subword_to_subid = params['subword_to_subid']
        self.new_wordvec(params['word_vecs'], params['word_vecs_o'], a=a, b=b, c=c)
      else:
        self.word_vecs = normalize(params['word_vecs'])
        