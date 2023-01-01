import numpy as np
import torch
from src.load import location, enc, mode, w2v
import gensim

from sklearn.model_selection import train_test_split
import sys, os
import re


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^\w(),\.!?\']", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string) 
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r"\'", "", string) #
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"\.", " . ", string) # 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9()\.,!?\']", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

def line_to_words(line, data):
  if data == 'sst1' or data == 'sst2':
    clean_line = clean_str_sst(line.strip())
  else:
    clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  words = words[1:]

  return words

def get_vocab(file_list, data, enc):
  max_sent_len = 0
  word_to_idx = {'[PADDING]':0}
  # Starts at 1 for padding
  idx = 1

  for filename in file_list:
    f = open(filename, "r", encoding = enc)
    for line in f:
        words = line_to_words(line, data)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

    f.close()

  return max_sent_len, word_to_idx

def CrossValidSplit(data):
  if 'test' not in data.keys():
      data['train'], data['test'], data['train_label'], data['test_label'] = train_test_split(data['train'], data['train_label'], test_size=.1, stratify=data['train_label'], random_state=1106)
  return data

##REAL
def load_data(data, padding=4):
  """
  Load training data (dev/test optional).
  """

  mode_lst = mode(data)
  dir_lst = [location(data, m) for m in mode_lst]
  encoding = enc(data)

  max_sent_len, word_to_idx = get_vocab(dir_lst, data, encoding)

  data_out = {}
  data_out['w2i'] = word_to_idx

  for mode_, dir_ in zip(mode_lst, dir_lst):
    text = []
    label = []
    
    with open(dir_, 'r', encoding=encoding) as f:
      for line in f:
        words = line_to_words(line, data)
        y = int(line.strip().split()[0])
        sent = [word_to_idx[word] for word in words]
        # end padding
        if len(sent) < max_sent_len:
            sent = sent + [0] * (max_sent_len - len(sent))
        # start padding
        sent = [0]*padding + sent

        text.append(sent)
        label.append(y)

    data_out[mode_] = torch.tensor(text).to(torch.int32)
    data_out[mode_ + '_label'] = torch.LongTensor(label)
  
  data_out = CrossValidSplit(data_out)
  
  return data_out

def build_wordvec(w2i, dim=300, var=.25):
  model = gensim.models.KeyedVectors.load_word2vec_format(w2v(), limit=500000,binary=True)

  vocab_size = len(w2i)
  wordvec = torch.FloatTensor(vocab_size, dim).uniform_(-var, var)
  
  ## i=0 for PADDING
  wordvec[0,:] = 0
  for w, i in w2i.items():
    if model.has_index_for(w):
      wordvec[i] = torch.tensor(model.get_vector(w)).to(torch.float)
  
  return wordvec
