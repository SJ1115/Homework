"""
&apos; -> '
##AT##-##AT## -> -
; -> (middle-sentence) .
&quot; -> "
&amp; -> &
&#91 -> [
&#93 -> ]

eng-train nospace
eng-test nospace
deu-train +space
deu-test +space
"""
import torch
import re
from src.load import url, loader
import torch

def clean_str_en(string):
    """
    Tokenization/String Clearing for English.
    """
    string = re.sub(r"&quot;", "\"", string)
    string = re.sub(r"&#91", "(", string)
    string = re.sub(r"&#93", ")", string)
    string = re.sub(r"##AT##-##AT##", "-", string)
    string = re.sub(r"&amp;", "&", string)
    string = re.sub(r"&apos;", "\'", string)
    string = re.sub(r";", ".", string)
    ## General string-cleaning
    string = string.lower()
    string = re.sub(r"[^\w(),\-:\.!?\'\"]", " ", string)
    string = re.sub(r"\'s", "s", string) 
    string = re.sub(r"\'ve", " have", string) 
    string = re.sub(r"won \'t", "will not", string) 
    string = re.sub(r"n \'t", " not", string) 
    string = re.sub(r"\'re", "re", string) 
    string = re.sub(r"\'d", "d", string) 
    string = re.sub(r"\'ll", "will", string)
    ## unify number to 'Num' toke
    string = re.sub(r'((-)?\d{1,3}(,\d{3})*(\.\d+)?)', 'N', string)
    string = re.sub(r'N{1,}', ' <Num> ', string)
    
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def clean_str_ge(string):
    """
    Tokenization/String Clearing for German.
    """
    string = re.sub(r"&quot;", "\"", string)
    string = re.sub(r"&#91", "(", string)
    string = re.sub(r"&#93", ")", string)
    string = re.sub(r"##AT##-##AT##", "-", string)
    string = re.sub(r"&amp;", "&", string)
    string = re.sub(r"&apos;", "\'", string)
    string = re.sub(r";", ".", string)
    ## General string-cleaning
    string = string.lower()
    string = re.sub(r"[^\w(),\-:\.!?\'\"]", " ", string)
    ## unify number to 'Num' toke
    string = re.sub(r'((-)?\d{1,3}(.\d{3})*(\,\d+)?)', 'N', string)
    string = re.sub(r'N{1,}', ' <Num> ', string)
    
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

def text_pair_to_words(text_e, text_d, max_length, min_length):
    if max_length <= min_length:
        ValueError("Max_Length must be larger than Min_Length")
    
    clean_text_e = clean_str_en(text_e)
    clean_text_d = clean_str_ge(text_d)
    
    words_e = clean_text_e.split(' ')
    words_d = clean_text_d.split(' ')

    len_e = len(words_e)
    len_d = len(words_d)
    if max(len_d, len_e) > max_length:
        return None, None
    elif min(len_d, len_e) < min_length:
        return None, None
    else:
        return words_e, words_d

def add_word(words, cur_id, word_to_id, table):
    for word in words:
        if not word in word_to_id:
            word_to_id[word] = cur_id
            table[word] = 1
            cur_id += 1
        else:
            table[word] += 1
    return cur_id, word_to_id, table

def cut_vocab(table, max_vocab):
    Val = list(table.values())
    Key = list(table.keys())
    if max_vocab:
        if max_vocab >= len(Val):
            return Key
        
        Val, Key = list(map(list, zip(*sorted(zip(Val, Key), key=lambda x: x[0], reverse=True))))
        threshold = Val[max_vocab]

        Key = [k for k, v in zip(Key, Val) if v >= threshold]

    return Key

def get_vocab(train_e, train_d, max_vocab, max_length, min_length):
    """
    max_vocab = -1 means no cut for vocab. But it doesn't mean not using <unk> token.
    """    
    eng_to_idx = {}
    ger_to_idx = {}
    
    eng_table = {}
    ger_table = {}
    
    idx_e = 0
    idx_d = 0

    for sent_e, sent_d in zip(train_e, train_d):
        words_e, words_d = text_pair_to_words(sent_e, sent_d, max_length, min_length)
        if words_e:
            idx_e, eng_to_idx, eng_table = add_word(words_e, idx_e, eng_to_idx, eng_table)
        if words_d:
            idx_d, ger_to_idx, ger_table = add_word(words_d, idx_d, ger_to_idx, ger_table)


    eng_words = cut_vocab(eng_table, max_vocab)
    ger_words = cut_vocab(ger_table, max_vocab)
        
    ## 0 for Padding
    eng_to_idx = {"<pad>" : 0, "<s>" : 1, "<e>": 2}
    ger_to_idx = {"<pad>" : 0, "<s>" : 1, "<e>": 2}
    for id, word in enumerate(eng_words):
        eng_to_idx[word] = id + 3
    for id, word in enumerate(ger_words):
        ger_to_idx[word] = id + 3
    return eng_to_idx, ger_to_idx

def line_pair_to_ids(lines_eng, lines_ger, eng_to_idx, ger_to_idx, get_unk, max_length, min_length):
    eng_out, ger_out = [], []
    eng_unk, ger_unk = len(eng_to_idx), len(ger_to_idx)

    for line_en, line_de in zip(lines_eng, lines_ger):
        words_en, words_de = text_pair_to_words(line_en, line_de, max_length, min_length)

        if words_en:
            if get_unk:
                words_en = [1] + [eng_to_idx[w] if w in eng_to_idx else eng_unk for w in words_en] + [2]
            else:
                words_en = [1] + [eng_to_idx[w] for w in words_en if w in eng_to_idx] + [2]
        if words_de:
            if get_unk:
                words_de = [1] + [ger_to_idx[w] if w in ger_to_idx else ger_unk for w in words_de] + [2]
            else:
                words_de = [1] + [ger_to_idx[w] for w in words_de if w in ger_to_idx] + [2]

        if words_en and words_de:
            eng_out.append(words_en)
            ger_out.append(words_de)
    
    return eng_out, ger_out

def pad_lines(lines, reverse, max_length):
    for line in lines:
        line += [0] * (max_length + 2 - len(line)) # +2 for <sos>&<eos>
        if reverse:
            line.reverse()
    return torch.tensor(lines)

def preprocess(src, max_vocab = 500000, get_unk = True, reverse = True, max_length=50, min_length=3):
    ## get data
    eng_train = loader(src.train.en, train=None)
    ger_train = loader(src.train.de, train=True)

    eng_dev = loader(src.dev.en, train=None)
    ger_dev = loader(src.dev.de, train=False)

    eng_test = loader(src.test.en[0], train=None) + loader(src.test.en[1], train=None)
    ger_test = loader(src.test.de[0], train=False) + loader(src.test.de[1], train=False)

    ## get vocab
    eng_to_idx, ger_to_idx = get_vocab(eng_train, ger_train, max_vocab, max_length, min_length)

    ## processing
    eng_train, ger_train = line_pair_to_ids(eng_train, ger_train, eng_to_idx, ger_to_idx, get_unk, max_length, min_length)
    eng_dev, ger_dev = line_pair_to_ids(eng_dev, ger_dev, eng_to_idx, ger_to_idx, get_unk, max_length, min_length)
    eng_test, ger_test = line_pair_to_ids(eng_test, ger_test, eng_to_idx, ger_to_idx, get_unk, max_length, min_length)

    ## padding
    eng_train = pad_lines(eng_train, reverse, max_length);  ger_train = pad_lines(ger_train, False, max_length) ## +1 for <S> token.
    eng_dev   = pad_lines(eng_dev, reverse, max_length);    ger_dev   = pad_lines(ger_dev, False, max_length)
    eng_test  = pad_lines(eng_test, reverse, max_length);   ger_test  = pad_lines(ger_test, False, max_length)

    ## additional processing
    if get_unk:
        eng_to_idx['<unk>'] = len(eng_to_idx)
        ger_to_idx['<unk>'] = len(ger_to_idx)

    return eng_to_idx, ger_to_idx, eng_train, ger_train, eng_dev, ger_dev, eng_test, ger_test
