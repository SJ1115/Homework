import numpy as np

def normalize(x, return_norm = False):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        norm = s.reshape((s.shape[0], 1))
        x /= norm + 1e-8
    elif x.ndim == 1:
        norm = np.sqrt((x * x).sum())
        x /= (norm + 1e-7)
    
    if return_norm:
        return x, norm
    else:
        return x

def sigmoid(x):
    x = np.maximum(-87, x)
    return 1 / (1 + np.exp(-x))

def get_word_vec(contexts, W, id_hash):      # target = (#target, ) ~= (6,)
    """return (c, H)"""
    sub_ids = []
    word_vecs = []
    for word in contexts:
        sub_index = id_hash[word]              # sub_index = (#sub, )
        sub_vector = W[sub_index]     # sub_vector = (#sub, D)
        sub_ids.append(sub_index)
        word_vecs.append(sub_vector.sum(axis=0))
    return np.array(word_vecs), sub_ids


def get_subword_vec(target, W, id_hash):       # center word = (1, )
    """return (n_subid, H)
    """
    sub_id = id_hash[target]
    sub_vec = W[sub_id].sum(axis=0)
    return sub_vec, np.array(sub_id)