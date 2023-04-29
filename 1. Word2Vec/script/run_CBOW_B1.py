import sys, os

sys.path.insert(0,'..')

import pickle
import copy
import numpy as np
import time
from tqdm import tqdm

from preprocess import *
from model import CBOW_B1






###### 데이터 읽기
with open(os.path.join(sys.path[0],"huffman_30K.pkl"), 'rb') as f:
    huff_30K = pickle.load(f)

with open(os.path.join(sys.path[0],"huffman_1B.pkl"), 'rb') as f:
    huff_1B = pickle.load(f)

with open(os.path.join(sys.path[0],"corpus_save.pkl"), 'rb') as f:
    corpusBig, corpus30K = pickle.load(f)

# 하이퍼파라미터 설정
window_size = 10
hidden_size = [50, 100, 300]
batch_size = 2048
learn_rate = .02
epoch_size = 3
word_size = [(4, 24), (7, 49), (13, 98), (25, 196), (50, 391), (100 ,782)]


# 파일 읽기용 설정
prefix = "30_K/new-"
suffix = "-of-100.pkl"

## word size별로 실험
for num_file, num_words in word_size:
    ## hidden은 한꺼번에
    model_list = []
    for hidden in hidden_size:
        # 실험시작
        model_list.append(CBOW_B1(True, hidden, huff_30K.codebook, huff_30K.nodebook))
    
    total_words = (num_words+1) * 500000 * epoch_size

    # 각 3에폭씩 학습
    word_cnt = 0
    lr = learn_rate

    timeiter = tqdm(range(num_file*epoch_size))

    for e in range(epoch_size):

        # 데이터 읽기: 파일별로 실행
        for i, num in enumerate(np.random.choice(range(100), 100, False)):
            #print(f"at hidden {hidden}, {e+1}-th epoch, learning file {i} started")
            run_start = time.time()

            with open(os.path.join(sys.path[0], prefix + f"{num:05}" + suffix), 'rb') as f:
                corpus_temp = pickle.load(f,)

            ######
            contexts, target = corpusBig.create_contexts_target(corpus_temp, window_size, False, False)
            
            
            for j, c, t in zip(range(len(target)), contexts, target):            
                
                for model in model_list:
                    model.learn(c, t, lr)

                if not j%batch_size:
                    l = [model.loss(t) for model in model_list]
                    timeiter.set_description(f"In {num_words}M, loss:{np.mean(l):.4f}")
                    lr = learn_rate * max(0.001, (1 - (word_cnt / total_words)))
                    word_cnt += batch_size
                    l = []

            timeiter.update(1)
            # learn in file END
#                temp = np.mean(l)
#                loss.append(temp)
#                print(f"{(time.time() - run_start) / 60:.2f}(m) needed. loss {temp:.4f}")

            if i == (num_file - 1): break
            ## word size에 맞춰 cut
        ## learning in an epoch End
    # learning of a model End

    # 나중에 사용할 수 있도록 필요한 데이터 저장
    for hidden, model in zip(hidden_size, model_list):
        word_vecs = model.word_vecs

        params = {}
        params['word_vecs'] = word_vecs.astype(np.float16)
        params['word_to_id'] = corpus30K.word_to_id
        params['id_to_word'] = corpus30K.id_to_word


        # or 'skipgram_params.pkl'
        with open(os.path.join(sys.path[0], "wordvec_res/" + "CBOW_" +
                f"{hidden}" + f"_{num_words}M.pkl"), 'wb') as f:
            pickle.dump(params, f, -1)
        print(f"File 'CBOW_{hidden}_{num_words}M.pkl saved")
# saving param of a model END
