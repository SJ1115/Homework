import numpy as np
import time
from src.vocab import Vocab
from src.func import sigmoid, get_word_vec, get_subword_vec, normalize



class CBOW:
    def __init__(self, hidden_size, Freq, n_samples = 5):
        '''I only implemented Neg-Sampling
        '''

        self.NS_freq = Freq
        self.num_samples = n_samples
        V = self.NS_freq[-1]+1

        #self.sign = -np.ones(self.num_samples+1)
        #self.sign[0] *= -1
        self.sign = np.zeros(self.num_samples+1, int)
        self.sign[0] = 1

        self.H = hidden_size

        # 가중치 초기화
        self.W_in = np.random.randn(V, self.H).astype('f') / np.sqrt(2 / (self.H))
        self.W_out = np.random.randn(V, self.H).astype('f') / np.sqrt(2 / (V))
        
        self.sig_slctd = None

        self.word_vecs = self.W_in

        #self.time = []
    
    def learn(self, contexts, target, alpha):
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        hidden = np.sum(self.W_in[contexts], axis=0) / len(contexts)
        #hidden = hidden.reshape(1, self.H)

        #### NS
        neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples)]
        tot_id = np.append([target], neg_id)

        W_slctd = self.W_out[tot_id]
        
        node_slctd = np.dot(W_slctd, hidden)
        
        self.sig_slctd = sigmoid(node_slctd)

        ## BACK

        #d_n_s =  alpha * self.sig_slctd * (1 - self.sig_slctd) * self.sign
        d_n_s =  alpha * (self.sig_slctd - self.sign)

        d_hidden = np.dot(d_n_s.reshape(1, -1), W_slctd).reshape(-1)
        d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
        
        
    ### hidden   : (H, )
    ### W_slctd  : (N+1, H)
    ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

        self.W_out[tot_id] -= d_W_s
            
        d_hidden /= len(contexts) 
        self.W_in[contexts] -= d_hidden

        return None
    
    def loss(self, target = None):
        """if the model is builted as Neg-Sampling methods,
            input 'target' is not necessory.
        """
        prop_slctd = np.array(list(map(lambda x, y: x if y else 1-x, self.sig_slctd, self.sign)))
        return -np.sum(np.log(prop_slctd + 1e-7))


class SkipGram:
    def __init__(self, hidden_size, Freq, n_samples = 5):
        '''HS = True 인 경우, Data에 huffman tree
                False인 경우, Data에 NS_Freq 
        '''

        self.NS_freq = Freq
        self.num_samples = n_samples
        V = self.NS_freq[-1]+1

        #self.sign =  - np.ones(self.num_samples+1, int)
        self.sign = np.zeros(self.num_samples+1, int)
        self.sign[0] = 1
        
        self.H = hidden_size

        # 가중치 초기화
        self.W_in = 1.42/np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        self.W_out = 1.42/np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        
        self.word_vecs = self.W_in
        self.sig_slctd = None
        #self.time = []
    
    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
        """
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        hidden = self.W_in[contexts].reshape(-1, self.H)
        
        #### NS
        neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples)]
        tot_id = np.append([target], neg_id)

        W_slctd = self.W_out[tot_id]
        
        node_slctd = np.dot(hidden, W_slctd.T)
        #node_slctd *= self.sign
        
        self.sig_slctd = sigmoid(node_slctd)

        #d_n_s =  alpha * (self.sig_slctd - 1) * self.sign
        #d_n_s =  alpha * self.sig_slctd * (1 - self.sig_slctd) * self.sign
        d_n_s =  alpha * (self.sig_slctd - self.sign) / self.num_samples
        
        d_hidden = np.dot(d_n_s, W_slctd)
        d_W_s = np.dot(d_n_s.T, hidden)
        
        ### hidden   : (H, )
        ### W_slctd  : (N+1, H)
        ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

        self.W_out[tot_id] -= d_W_s
        
        #d_hidden /= len(contexts)
        self.W_in[contexts] -= d_hidden.squeeze()

        return None

    def loss(self, ):
        """Since model is built as Neg_sampling,
            you don't have to input 'target'."""
        #return np.sum(np.log(self.sig_slctd + 1e+7))
        prop_slctd = np.array(list(map(lambda x, y: x if y else 1-x, self.sig_slctd, self.sign)))
        return -np.sum(np.log(prop_slctd + 1e-7))

class SkipGram_SI:
    def __init__(self, id_to_subid, vocab_size, hidden_size, NS_freq, n_samples = 5):
        '''HS = True 인 경우, Data에 codebook / sub에 nodebook
                False인 경우, Data에 NS_Freq  / sub에 ns_size
        '''
        self.id_hash = id_to_subid

        self.NS_freq = NS_freq
        self.num_samples = n_samples
        V = len(id_to_subid)
        SV = vocab_size
        # SV = sub-vocab-size

        self.sign = np.zeros(self.num_samples+1, int)
        self.sign[0] = 1
        
        self.H = hidden_size

        # 가중치 초기화(Xavier initialization)
        self.W_in = np.random.randn(SV, self.H).astype('f') * np.sqrt(2 / (self.H + SV))
        
        self.W_out = np.random.randn(V, self.H).astype('f') * np.sqrt(2 / (self.H + V))
        

        #self.word_vecs = (self.W_in_ori, self.W_in_sub)
        self.sig_slctd = None
        #self.time = []
    
    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
            We only implemented Negative Sample Model.
        """

        hidden, subid_target = get_subword_vec(target, self.W_in, self.id_hash)
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        
        neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples*len(contexts))]
        
        #sub_cont_id = [self.id_hash[c] for c in contexts]
        #tot_id = np.append([sub_cont_id], neg_id)
        tot_id = np.append([contexts], neg_id)

        W_slctd = self.W_out[tot_id]

        node_slctd = np.dot(hidden, W_slctd.T)
        
        n_sign = np.repeat(self.sign, len(contexts))
            
        self.sig_slctd = sigmoid(node_slctd)

        d_n_s =  alpha * (self.sig_slctd - n_sign)
            
        d_hidden = np.dot(d_n_s, W_slctd)

        d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N+1, H)
        ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension
        
        self.W_out[tot_id] -= d_W_s
        

        self.W_in[subid_target] -= d_hidden

        return None

    def loss(self, target = None):
        """if teh model is built as Neg_sampling,
            you don't have to input 'target'."""
        prop_slctd = np.array(list(map(lambda x, y: x if y else 1-x, self.sig_slctd, self.sign)))
        return -np.sum(np.log(prop_slctd + 1e-7)) #/ len(self.id_hash[target])

class SkipGram_SISI:
    def __init__(self, id_to_subid, vocab_size, hidden_size, NS_freq, n_samples = 5):
        '''HS = True 인 경우, Data에 codebook / sub에 nodebook
                False인 경우, Data에 NS_Freq  / sub에 ns_size
        '''
        self.id_hash = id_to_subid

        self.NS_freq = NS_freq
        self.num_samples = n_samples
        V = len(id_to_subid)
        SV = vocab_size
        # SV = sub-vocab-size

        self.H = hidden_size

        # 가중치 초기화(Xavier initialization)
        self.W_in = np.random.randn(SV, self.H).astype('f') * np.sqrt(2 / (self.H + SV))
        
        self.W_out = np.random.randn(SV, self.H).astype('f') * np.sqrt(2 / (self.H + SV))
        

        #self.word_vecs = (self.W_in_ori, self.W_in_sub)
        self.sig_slctd = [None, None]
        #self.time = []
    def _learn(self, alpha, subid, hidden, subid_target, Pos=1):
        W_slctd = self.W_out[subid]

        node_slctd = np.dot(hidden, W_slctd.T)
            
        self.sig_slctd[Pos] = sigmoid(node_slctd)

        d_n_s =  alpha * (self.sig_slctd - Pos)
            
        d_hidden = np.dot(d_n_s, W_slctd)

        d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N+1, H)
        ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension
        
        self.W_out[subid] -= d_W_s
        
        self.W_in[subid_target] -= d_hidden
        
        return None


    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
            We only implemented Negative Sample Model.
        """

        hidden, subid_target = get_subword_vec(target, self.W_in, self.id_hash)
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        
        neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples*len(contexts))]
        
        #sub_cont_id = [self.id_hash[c] for c in contexts]
        #tot_id = np.append([sub_cont_id], neg_id)
        
        neg_subid = [self.id_hash[i] for neg in neg_id for i in neg ]
        pos_subid = [self.id_hash[i] for i in contexts]

        self._learn(alpha, neg_subid, hidden, subid_target, Pos=0)
        self._learn(alpha, pos_subid, hidden, subid_target, Pos=1)

        return None

    def loss(self, target = None):
        """if teh model is built as Neg_sampling,
            you don't have to input 'target'."""
        pos_slctd = self.sig_slctd[1]
        neg_slctd = 1 - self.sig_slctd[0]
        return -(np.sum(np.log(pos_slctd + 1e-7)) + np.sum(np.log(pos_slctd + 1e-7))) #/ len(self.id_hash[target])



class SkipGram_SI_New:
    def __init__(self, id_to_subid, vocab_size, hidden_size, NS_freq, n_samples = 5):
        '''Subword와 word를 동시에 사용해보자.
        '''
        self.id_hash = id_to_subid

        self.NS_freq = NS_freq
        self.num_samples = n_samples
        V = len(id_to_subid)
        SV = vocab_size
        # SV = sub-vocab-size

        self.sign = np.zeros(self.num_samples+1, int)
        self.sign[0] = 1
        
        self.H = hidden_size

        # 가중치 초기화(Xavier initialization)
        self.W_in = np.random.randn(SV, self.H).astype('f') * np.sqrt(2 / (self.H + SV))
        self.W_in_o = np.random.randn(V, self.H).astype('f') * np.sqrt(2 / (self.H + V))

        self.W_out = np.random.randn(V, self.H).astype('f') * np.sqrt(2 / (self.H + V))
        

        #self.word_vecs = (self.W_in_ori, self.W_in_sub)
        self.sig_slctd = None
        #self.time = []
    
    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
            We only implemented Negative Sample Model.
        """

        hidden, subid_target = get_subword_vec(target, self.W_in, self.id_hash)
        hidden += self.W_in_o[target]
        hidden /= (len(subid_target)+1)
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        
        neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples*len(contexts))]
        
        #sub_cont_id = [self.id_hash[c] for c in contexts]
        #tot_id = np.append([sub_cont_id], neg_id)
        tot_id = np.append([contexts], neg_id)

        W_slctd = self.W_out[tot_id]

        node_slctd = np.dot(hidden, W_slctd.T)
        
        n_sign = np.repeat(self.sign, len(contexts))
            
        self.sig_slctd = sigmoid(node_slctd)

        d_n_s =  alpha * (self.sig_slctd - n_sign)
            
        d_hidden = np.dot(d_n_s, W_slctd)

        d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N+1, H)
        ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

        self.W_out[tot_id] -= d_W_s
        
        d_hidden /= (len(subid_target)+1)
        self.W_in_o[target] -= d_hidden
        self.W_in[subid_target] -= d_hidden 

        return None

    def loss(self, target = None):
        """if teh model is built as Neg_sampling,
            you don't have to input 'target'."""
        prop_slctd = np.array(list(map(lambda x, y: x if y else 1-x, self.sig_slctd, self.sign)))
        return -np.sum(np.log(prop_slctd + 1e-7)) #/ len(self.id_hash[target])


###################################################
### Below Here is for Text Classification Model ###
###################################################


class TextClassifier:
    def __init__(self, vocab_size, huffman ,hidden = 10):
        """The Model is structural the same with CBOW, with Hierarchical Softmax.
        Difference comes from the target of CBOW is 'word', but is 'class' in TC."""
        self.codebook = huffman.codebook
        self.nodebook = huffman.nodebook

        ## ind 0 for Unknown
        V = vocab_size + 1
        self.O = len(self.codebook)
        H = hidden

        1.42/np.sqrt(V + H)
        1.42/np.sqrt(self.O-1 + H)

        self.W_in = 1.42/np.sqrt(V + H) * np.random.uniform(low=-1, size=(V, H)).astype('f')
        self.W_in[0] = 0
        self.W_out = 1.42/np.sqrt(self.O-1 + H) * np.random.uniform(low= -1, size=(self.O-1, H)).astype('f')
        
        self.sig_slctd = None
        self.params = [self.W_in, self.W_out]
        return

    def learn(self, sentence, label, lr):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
        """
        hidden = np.sum(self.W_in[sentence], axis=0)
        #hidden, norm = normalize(hidden, return_norm = True) 
        hidden /= len(sentence)
        #hidden = hidden.reshape(1, self.H)
            
        W_slctd = self.W_out[self.nodebook[label]]
        
        node_slctd = np.dot(W_slctd, hidden)
        
        self.sig_slctd = sigmoid(node_slctd)

        #### BackProp
        
        d_n_s = lr * (self.sig_slctd + self.codebook[label] - 1)
        
        d_hidden = np.dot(d_n_s.reshape(1, -1), W_slctd).reshape(-1)
        d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N, H)
        ### d_n_s    : (N, ) N은 node 수, H는 hidden dimension

        self.W_out[self.nodebook[label]] -= d_W_s

        d_hidden /= len(sentence)    
        #d_hidden /= norm

        self.W_in[sentence] -= d_hidden

        return None
    
    def loss(self, label):
        prop_slctd = np.array(list(map(lambda x, y: 1-x if y else x, self.sig_slctd, self.codebook[label])))
        return -np.sum(np.log(prop_slctd + 1e-7))

    def predict(self, sentence, return_prop = False):
        """Since the model uses Hierarchical Softmax, It is Sub-Efficient when we calculate real softmax
        """
        hidden = np.sum(self.W_in[sentence], axis=0)
        #hidden = normalize(hidden) 
        hidden /= len(sentence)
        #hidden = hidden.reshape(1, self.H)
        
        hidden = np.dot(self.W_out, hidden)
        
        hidden = sigmoid(hidden)
        #### 
        prop = np.empty(self.O, dtype=np.float32)
        
        for i, node in self.nodebook.items():
            code = self.codebook[i]
            p = [1-hidden[n] if c else hidden[n] for n, c in zip(node, code)]   
            prop[i] = np.product(p)
        if return_prop:
            return prop
        else:
            return np.argmax(prop)
    
    def test(self, test, test_label):
        correct = 0
        for x, y in zip(test, test_label):
            correct += self.predict(x) == y
        
        return correct / len(test_label)