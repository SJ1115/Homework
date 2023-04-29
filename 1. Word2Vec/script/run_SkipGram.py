import sys
sys.path.insert(0, "/data/user6/LSJ/freshman/DL_Scratch_RFR")
import pickle
from DL_Scratch_RFR.common.trainer import Trainer
from DL_Scratch_RFR.common.optimizer import *
from DL_Scratch_RFR.common.util import create_contexts_target
from DL_Scratch_RFR.dataset import ptb

import copy
import numpy as np
import time

from preprocess import huffman
from model import SkipGram_HH

######
# 데이터 읽기
with open("/data/user6/LSJ/freshman/30K_huffman.pkl", 'rb') as f:
    huff_0 = pickle.load(f)

with open("/data/user6/LSJ/freshman/word_list_30K", 'rb') as f:
    word_list = pickle.load(f)
######
new_codebook = {}
new_nodebook = {}
 
# getting all values of first dictionary
for key, val in word_list['id_to_word'].items():
    # getting result with default value list and extending
    # according to value obtained from get()
    new_codebook.setdefault(key, []).extend(huff_0.codebook.get(val, []))
    new_nodebook.setdefault(key, []).extend(huff_0.nodebook.get(val, []))

huff_30K = copy.copy(huff_0)
huff_30K.nodebook = new_nodebook
huff_30K.codebook = new_codebook

######

# 하이퍼파라미터 설정
vocab_size = len(huff_0.codebook.keys())
window_size = 5
hidden_size = [300, 300, 600]
batch_size = 200
learn_rate = [.025, .017, .008]

# 파일 읽기용 설정
prefix = "/data/user6/LSJ/freshman/30_K/"
suffix = "-of-100.pkl"

## hidden별로
for run_id, hidden in enumerate(hidden_size):
    model = SkipGram_HH(vocab_size, hidden, window_size, None, huff_30K)        
    
    # 각 3에폭씩 학습
    for e, lr in enumerate(learn_rate):
        optimizer = SGD(lr=lr)
        
        word_size = 0
        # 데이터 읽기: 파일별로 실행
        for i in range(150):
            print(f"at hidden {hidden}, {e+1}-th epoch, learning file {i} started")
            run_start = time.time()

            with open(prefix + f"{i:05}" + suffix, 'rb') as f:
                corpus_temp = pickle.load(f,)

            word_size += len(corpus_temp)

            ######
            contexts, target = create_contexts_target(corpus_temp, window_size)

            # 모델 등 생성
            trainer = Trainer(model, optimizer)

            # 학습 시작
            trainer.fit(contexts, target, 1, batch_size, vervose = False)
            # learn in file END
            print(f"{(time.time() - run_start) / 60:.2f}(m) needed.")

            if run_id in (1, 2):
                if word_size > 94200000:
                    break
        # learn a Epoch END
    # learning of a model END

    # 나중에 사용할 수 있도록 필요한 데이터 저장

    word_vecs = model.word_vecs

    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_list['word_to_id']
    params['id_to_word'] = word_list['id_to_word']



        # or 'skipgram_params.pkl'
    with open("/data/user6/LSJ/freshman/wordvec_res/" + "SG_" + f"{hidden}" + 
            f"_{int(word_size/1000000)}M.pkl", 'wb') as f:
        pickle.dump(params, f, -1)
    # saving param of a model ENDimport sys
