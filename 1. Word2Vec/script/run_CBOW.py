import sys, os

sys.path.insert(0,'..')

import pickle
import copy
import numpy as np
import time
from tqdm import tqdm

from preprocess import *
from model import CBOW_B1
from W2Vtest import SemanticSyntacticTest as SST

###### 데이터 읽기
#with open(os.path.join(sys.path[0],"huffman_30K.pkl"), 'rb') as f:
#    huff_30K = pickle.load(f)

with open(os.path.join(sys.path[0],"huffman_full.pkl"), 'rb') as f:
    huff_1B = pickle.load(f)

with open(os.path.join(sys.path[0],"corpus_new.pkl"), 'rb') as f:
    corpusBig = pickle.load(f)


######

corpusBig.threshold = {}
corpusBig.set_subsample()
######

# 하이퍼파라미터 설정
window_size = 5
hidden_size = [300, 640]
batch_size = 1024
epoch_size = 3
learn_rate = .02

# 파일 읽기용 설정
prefix = "1_B/new-"
suffix = "-of-100.pkl"

codebook = {}
nodebook = {}
for i, word in corpusBig.id_to_word.items():
    codebook[i] = np.array(huff_1B.codebook[word])
    nodebook[i] = np.array(huff_1B.nodebook[word])

## hidden별로
model = [CBOW_B1(True, hidden, codebook, nodebook) for hidden in hidden_size]


total_words = 600000000 * epoch_size
word_cnt = 0
l0 = []; lr = learn_rate

timeiter = tqdm(total = int(epoch_size * 100))
# 각 3에폭씩 학습
for e in range(epoch_size):

    # 데이터 읽기: 파일별로 실행
    for i, num in enumerate(np.random.choice(range(100), 100, False)):

        with open(os.path.join(sys.path[0], prefix + f"{num:05}" + suffix), 'rb') as f:
            corpus_temp = pickle.load(f,)

        ######
        contexts, target = corpusBig.create_contexts_target(corpus_temp, window_size, False, True)
        contexts, target = mix(contexts, target, False)
        del corpus_temp

        # 학습 시작
        for j, c, t in zip(range(len(target)), contexts, target):
            
            for m in model:
                m.learn(c, t, lr)
            
            lr = learn_rate *  max(0.001, 1 - (word_cnt / total_words))
            word_cnt += 1

            if not j%batch_size:
                string = f"loss:{sum([m.loss(t) for m in model]):.4f} at {i}-th in {e}"
                timeiter.set_description(string)

                
        timeiter.update(1)

        
    # learn a Epoch END
# learning of a model END
for hidden, m in zip(hidden_size, model):
    params = {}
    params['word_vecs'] = m.word_vecs.astype(np.float16)
    params['word_to_id'] = corpusBig.word_to_id
    params['id_to_word'] = corpusBig.id_to_word

    with open(os.path.join(sys.path[0], f"wordvec_res/CBOW_{hidden}_1B.pkl"), 'wb') as f:
        pickle.dump(params, f,)

    #print(f"Test for HS_{hidden_size}_1B")
    print(f"Test for CBOW_{hidden_size}_1B")
    sst = SST(params['word_to_id'], params['id_to_word'], params['word_vecs'])
    sst.test_0()

    print(f"total {sum(sst.result) / len(sst.result)}")
    print(f"seman {sum(sst.result_sem) / len(sst.result_sem)}")
    print(f"synta {sum(sst.result_syn) / len(sst.result_syn)}")

    # 나중에 사용할 수 있도록 필요한 데이터 저장

"""    word_vecs = model.word_vecs

    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_list['word_to_id']
    params['id_to_word'] = word_list['id_to_word']



        # or 'skipgram_params.pkl'
    with open("/data/user6/LSJ/freshman/wordvec_res/" + "CBOW_" + f"{hidden}" + 
            f"_{int(word_size/100000)}M.pkl", 'wb') as f:
        pickle.dump(params, f, -1)"""
    # saving param of a model END