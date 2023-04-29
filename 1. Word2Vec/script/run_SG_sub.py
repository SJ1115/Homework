import sys, os

sys.path.insert(0,'..')

import pickle
import copy
import numpy as np
from tqdm import tqdm

from preprocess import *
from model import SkipGram_B1, SkipGram_B2
from W2Vtest import SemanticSyntacticTest as SST

######
# 데이터 읽기
with open(os.path.join(sys.path[0],"huffman_1B.pkl"), 'rb') as f:
    huff = pickle.load(f)

with open(os.path.join(sys.path[0],"corpus_save.pkl"), 'rb') as f:
    corpusBig, corpus30K = pickle.load(f)

with open(os.path.join(sys.path[0],"NSTable_1B copy.pkl"), 'rb') as f:
    table = pickle.load(f)

del corpus30K


corpusBig.threshold = {}
corpusBig.set_subsample()


######

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 300
neg_samples = [5, 15]
batch_size = 1024
epoch_size = 3
learn_rate = .02

# 파일 읽기용 설정
prefix = "1_B/"
suffix = "-of-100.pkl"

## hidden별로
for neg in neg_samples:
    #model = SkipGram_B2(True, hidden_size, huff.codebook, huff.nodebook)
    model = SkipGram_B2(False, hidden_size, table, neg)
    #model = SkipGram_B1(False, hidden_size, table, neg)
    
    total_words = 371397491 * epoch_size
    word_cnt = 0
    l0 = []; lr = learn_rate

    timeiter = tqdm(total = epoch_size * 100)
    # 각 3에폭씩 학습
    for e in range(epoch_size):

        # 데이터 읽기: 파일별로 실행
        for i, num in enumerate(np.random.choice(range(100), 100, False)):

            with open(os.path.join(sys.path[0],prefix + f"{num:05}" + suffix), 'rb') as f:
                corpus_temp = pickle.load(f,)

            ######
            contexts, target = corpusBig.create_contexts_target(corpus_temp, window_size, True, False)

            # 학습 시작
            for j, c, t in zip(range(len(target)), contexts, target):

                model.learn(c, t, lr)

                if not j%batch_size:
                    #string = f"sub_HS, loss: {model.loss(t):.4f} at {i + 100*e}" 
                    string = f"sub_NS_{neg}, loss: {model.loss(t):.4f} at {i + 100*e}" 
                    timeiter.set_description(string)
                    
                    lr = learn_rate * max(0.0001, (1 - (word_cnt / total_words)))
                    word_cnt += batch_size
                    #l0 = []



            params = {}
            params['word_vecs'] = model.word_vecs.astype(np.float16)
            params['word_to_id'] = corpusBig.word_to_id
            params['id_to_word'] = corpusBig.id_to_word

            #with open(os.path.join(sys.path[0], f"wordvec_res/sub/HS_at_{e}_{hidden_size}.pkl"), 'wb') as f:
            with open(os.path.join(sys.path[0], f"wordvec_res/sub/NS_{neg}_at_{e}_{hidden_size}.pkl"), 'wb') as f:
                pickle.dump(params, f,)
            timeiter.update(1)
        # learn a Epoch END
    # learning of a model END
    timeiter.clear()
    
    # 나중에 사용할 수 있도록 필요한 데이터 저장
    params = {}
    params['word_vecs'] = model.word_vecs.astype(np.float16)
    params['word_to_id'] = corpusBig.word_to_id
    params['id_to_word'] = corpusBig.id_to_word
    
    # Test ofr result
    #with open(os.path.join(sys.path[0], f"wordvec_res/sub/HS_{hidden_size}_1B.pkl"), 'wb') as f:
    with open(os.path.join(sys.path[0], f"wordvec_res/sub/NS_{neg}_{hidden_size}_1B.pkl"), 'wb') as f:
        pickle.dump(params, f,)

    #print(f"Test for HS_{hidden_size}_1B")
    print(f"Test for NS_{neg}_{hidden_size}_1B")
    sst = SST(params['word_to_id'], params['id_to_word'], params['word_vecs'])
    sst.test_0()

    print(f"total {sum(sst.result) / len(sst.result)}")
    print(f"seman {sum(sst.result_sem) / len(sst.result_sem)}")
    print(f"synta {sum(sst.result_syn) / len(sst.result_syn)}")

"""    word_vecs = model.word_vecs

    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_list['word_to_id']
    params['id_to_word'] = word_list['id_to_word']



        # or 'skipgram_params.pkl'
    with open("/data/user6/LSJ/freshman/wordvec_res/" + "CBOW_" + f"{hidden}" + 
            f"_{int(word_size/100000)}M.pkl", 'wb') as f:
        pickle.dump(params, f, -1)"""
    # saving param of a model ENDimport sys
