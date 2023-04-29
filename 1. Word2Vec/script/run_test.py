import sys, os
sys.path.insert(0, "..")

from W2Vtest import SemanticSyntacticTest as SST
import pickle

print("CHECK for 'CB_640_Fin'")
with open(os.path.join(sys.path[0],"wordvec_res/CBOW_640_1B.pkl"), 'rb') as f:
    param = pickle.load(f,)

#print("CHECK for sub_SG_HS_300_Fin")
#with open(os.path.join(sys.path[0],"wordvec_res/sub/HS_300_1B.pkl"), 'rb') as f:
#    param = pickle.load(f,)

#print("CHECK for SG_at_2_640ep.2")
#with open(os.path.join(sys.path[0],"wordvec_res/vec__SG_at_2_640.pkl"), 'rb') as f:
#    param = pickle.load(f,)

#print("CHECK for 'CBOW_600_391M'")
#with open(os.path.join(sys.path[0],"wordvec_res/CBOW_50_782M.pkl"), 'rb') as f:
#    param = pickle.load(f,)

from time import time
s = time()
sst = SST(param['word_to_id'], param['id_to_word'], param['word_vecs'])
print(f"{time() - s:.4} sec for normalizing")

sst.test_0()


print(f"total {sum(sst.result) / len(sst.result)}")
print(f"seman {sum(sst.result_sem) / len(sst.result_sem)}")
print(f"synta {sum(sst.result_syn) / len(sst.result_syn)}")
"""
hidden = [50, 100, 300, 600]
words = [391]
for w in words:
    for h in hidden:
        print(f"CHECK for 'CBOW_{h}_{w}M'")
        with open(os.path.join(sys.path[0], f"wordvec_res/CBOW_{h}_{w}M.pkl"), 'rb') as f:
            param = pickle.load(f,)
        from time import time
        s = time()
        sst = SST(param['word_to_id'], param['id_to_word'], param['word_vecs'])
        print(f"{time() - s:.4} sec for normalizing")

        sst.test_0()


        print(f"total {sum(sst.result) / len(sst.result)}")
        print(f"seman {sum(sst.result_sem) / len(sst.result_sem)}")
        print(f"synta {sum(sst.result_syn) / len(sst.result_syn)}")"""