import sys, os

sys.path.insert(0,'..')

import pickle
import numpy as np
import time
from tqdm import tqdm

from src.vocab import Vocab
from src.model import *
from src import preprocess
from src.func import normalize
from src.WStest import wordSimTest, SemanticSyntacticTest
######
# 데이터 가져오기
from datasets import load_dataset

wikidata = load_dataset("wikipedia", "20220301.en")

max_words = 1000000
max_words = 150000000
min_count = 5
split_size = 100
## for Memory Burdon, we use subset of corpus when
##  feeding the model


# 150M Words 까지만 데이터 수집
vocab = Vocab(range(2, 6))
corpus = [];    i = 0;   n = 0
timeiter = tqdm(total = max_words, desc="Collecting Documents...")
while True:

    doc = wikidata['train'][i]['text']

    lines, k = vocab.add_corpus(doc) 

    corpus += lines
    n += k
    i += 1
    timeiter.update(k)
    if n > max_words:
        break

del wikidata
print(f"To Collect {n} words, scrapped {i} documents")
# 데이터 전처리 및 Tokenize 준비
corpus = vocab.reduce_corpus(corpus, min_count)
vocab.get_from_corpus()

vocab.get_freq_list()

vocab.set_subsample(1e-4)

# Split for Memory Burdon
corpus = np.array_split(np.array(corpus, dtype=object), split_size)
# 하이퍼파라미터 설정

hidden_size = 300
epoch_size = 3
batch_size = 2048


model = SkipGram_SI_New(vocab.id_to_subid, vocab.sub_vocab_size, hidden_size, vocab.freq_list, 5)
#model = SkipGram(hidden_size, vocab.freq_list, 5)
#model = CBOW(hidden_size, vocab.freq_list, 5)

if model.__class__.__name__ == 'SkipGram_SI_New':
    model_flag = "SISG"
    learn_rate = .05
elif model.__class__.__name__ == 'SkipGram':
    model_flag = "SG"
    learn_rate = .02
else:
    model_flag = "CBOW"
    learn_rate = .02


lr = learn_rate; learn_count = 0

timeiter = tqdm(total = split_size*epoch_size)

for epoch in range(epoch_size):
    for corp_i in np.random.choice(range(split_size), split_size, False):
        context, target = vocab.create_contexts_target(corpus[corp_i], window_size=5, subsampling=True, vervose=False)
        context, target = preprocess.mix(context, target, vervose=False)
        
        loss = []

        for i, (c, t) in enumerate(zip(context, target)):
            model.learn(c, t, lr)

            lr = learn_rate * max(0.0001, (1 - learn_count/(epoch_size*n)))
            learn_count += 1
            if not i % batch_size:  
                

                l = model.loss()
                loss.append(l)
                #timeiter.update(batch_size)
                timeiter.set_description(f"loss : {np.mean(loss): .4f}")



            #with open(os.path.join(sys.path[0], f"wordvec_res/sub/HS_at_{e}_{hidden_size}.pkl"), 'wb') as f:
            #with open(os.path.join(sys.path[0], f"wordvec_res/sub/NS_{neg}_at_{e}_{hidden_size}.pkl"), 'wb') as f:
            #    pickle.dump(params, f,)
        # learn a Epoch END
        timeiter.update(1)
# learning of a model END
timeiter.clear()

model.W_in = model.W_in.astype(np.float16)

if model_flag == 'SISG':
    model.W_in_o = model.W_in_o.astype(np.float16)
    vocab.build_wordvec(model.W_in, model.W_in_o)
else:
    vocab.word_vecs = normalize(model.W_in)
    # 나중에 사용할 수 있도록 필요한 데이터 저장
params = {}
params['word_vecs'] = model.W_in
params['word_to_id'] = vocab.word_to_id
params['id_to_word'] = vocab.id_to_word
if model_flag == 'SISG':
    params['word_vecs_o'] = model.W_in_o
    params['id_to_subid'] = vocab.id_to_subid
    params['subid_to_subword'] = vocab.subid_to_subword
    params['subword_to_subid'] = vocab.subword_to_subid

# Test ofr result
with open(os.path.join(sys.path[0], f"result/{model_flag}_{hidden_size}_ENWK{int(max_words / 1000000)}M.pkl"), 'wb') as f:
    pickle.dump(params, f,)

#print(f"Test for HS_{hidden_size}_1B")
mode = ['WS353', 'RW']

if model_flag == "SISG":
    sub_inf = [True, False]
else:
    sub_inf = [False]

for m in mode:
    for s in sub_inf:
        print(f"WordSim_Test for {model_flag}_{hidden_size}, mode={m}, SI={s}")
        sst = wordSimTest(vocab, m, s)
        print(sst)

sst = SemanticSyntacticTest(vocab)

sst.test_0()

print(f"SemSyn_Test for {model_flag}_{hidden_size}")
print(f"total {sum(sst.result) / len(sst.result)}")
print(f"seman {sum(sst.result_sem) / len(sst.result_sem)}")
print(f"synta {sum(sst.result_syn) / len(sst.result_syn)}")


