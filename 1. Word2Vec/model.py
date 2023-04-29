import numpy as np
import time

class CBOW_B1:
    def __init__(self, HS, hidden_size, Data, Data_sub = 5):
        ''' HS = True 인 경우, Data에 huffman
                 False인 경우, Data에 NS_freq, num_samples
        '''
        self.HS = HS

        if self.HS:
            self.codebook = Data
            self.nodebook = Data_sub
            V = len(self.codebook)
        else:
            self.NS_freq = Data
            self.num_samples = Data_sub
            V = self.NS_freq[-1]+1

            self.sign = -np.ones(self.num_samples+1)
            self.sign[0] *= -1

        self.H = hidden_size
        self.vervose = False

        # 가중치 초기화
        W_in = 1.42 /np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        if HS:
            W_out = 1.42 /np.sqrt(V-1 + self.H) * np.random.randn(V-1, self.H).astype('f')
        else:
            W_out = 1.42 /np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        
        self.W_in, self.W_out = W_in, W_out
        
        self.sig_slctd = None


        self.word_vecs = W_in

        #self.time = []
    
    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
        """
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]

        hidden = np.sum(self.W_in[contexts], axis=0) / len(contexts)
        #hidden = hidden.reshape(1, self.H)

        if self.HS: #### HS
            
            #nodes = np.array(self.huffman.nodebook[target])
            #codes = np.array(self.huffman.codebook[target])

            W_slctd = self.W_out[self.nodebook[target]]
            
            node_slctd = np.dot(W_slctd, hidden)
            
            self.sig_slctd = 1 / (1 + np.exp(-node_slctd))

            #### BackProp
            
            d_n_s = alpha * (self.sig_slctd + self.codebook[target] - 1)
            
            d_hidden = np.dot(d_n_s.reshape(1, -1), W_slctd).reshape(-1)
            d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N, H)
        ### d_n_s    : (N, ) N은 node 수, H는 hidden dimension

            self.W_out[self.nodebook[target]] -= d_W_s
            

        else: #### NS
            neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples)]
            
            W_slctd = self.W_out[np.append([target], neg_id)]
            
            node_slctd = np.dot(W_slctd, hidden)
            node_slctd *= self.sign
            
            self.sig_slctd = 1 / (1 + np.exp(-node_slctd))

            ## BACK

            d_n_s =  alpha * self.sig_slctd * (1 - self.sig_slctd) * self.sign
            
            d_hidden = np.dot(d_n_s.reshape(1, -1), W_slctd).reshape(-1)
            d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
            
        ### hidden   : (H, )
        ### W_slctd  : (N+1, H)
        ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

            self.W_out[np.append([target], neg_id)] -= d_W_s
            
        #d_hidden /= len(contexts) ?????
        self.W_in[contexts] -= d_hidden

        return None
    
    def loss(self, target = None):
        """if the model is builted as Neg-Sampling methods,
            input 'target' is not necessory.
        """
        if self.HS:
            prop_slctd = np.array(list(map(lambda x, y: 1-x+1e-7 if y else x + 1e-7, self.sig_slctd, self.codebook[target])))
            return -np.sum(np.log(prop_slctd))
        else:
            return -np.sum(np.log(self.sig_slctd + 1e-7))

class SkipGram_B1:
    def __init__(self, HS, hidden_size, Data, Data_sub = 5):
        '''HS = True 인 경우, Data에 huffman tree
                False인 경우, Data에 NS_Freq 
        '''
        self.HS = HS

        if self.HS:
            self.codebook = Data
            self.nodebook = Data_sub
            V = len(self.codebook)
        else:
            self.NS_freq = Data
            self.num_samples = Data_sub
            V = self.NS_freq[-1]+1

            #self.sign = -np.ones(self.num_samples+1)
            #self.sign[0] *= -1
            self.sign = -np.ones(self.num_samples, int)
            
        
        self.H = hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, self.H).astype('f')
        if HS:
            W_out = 0.01 * np.random.randn(V-1, self.H).astype('f')
        else:
            W_out = 0.01 * np.random.randn(V, self.H).astype('f')
        
        self.params = [W_in, W_out]
        self.sig_slctd = None

        self.word_vecs = W_in

        #self.time = []
    
    def learn(self, contexts, target, alpha):
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        hidden = self.params[0][target]

        if self.HS: #### HS
            d_hidden = np.zeros_like(hidden)
            for con in contexts:
                #nodes = np.array(self.huffman.nodebook[con])
                #codes = np.array(self.huffman.codebook[con])

                #W_slctd = self.params[1][self.nodebook[con]]
                
                node_slctd = np.dot(self.params[1][self.nodebook[con]], hidden)
                self.sig_slctd = 1 / (1 + np.exp(-node_slctd))

                #prop_slctd = np.array(list(map(lambda x, y: 1-x+1e-7 if y else x + 1e-7, sig_slctd, self.codebook[con])))
                #loss -= np.sum(np.log(prop_slctd))
                
                #### BackProp
                
                d_n_s = alpha * (self.codebook[con] -1 + self.sig_slctd)
                
                d_hidden += np.dot(d_n_s.reshape(1, -1), self.params[1][self.nodebook[con]]).reshape(-1)
                d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
                
            ### hidden   : (H, )
            ### W_slctd  : (N, H)
            ### d_n_s    : (N, ) N은 node 수, H는 hidden dimension
                #print(d_W_s.shape)
                self.params[1][self.nodebook[con]] -= d_W_s
                #self.grads[1][nodes] += d_W_s
                
        else: #### NS
            #neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), len(contexts) * self.num_samples)]
            neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples)]
            tot_id = np.append(contexts, neg_id)


            W_slctd = self.params[1][tot_id]

            #n_sign = np.repeat(self.sign, len(contexts))
            n_sign = np.append(np.ones_like(contexts, int), len(contexts)*self.sign)
            
            node_slctd = np.dot(W_slctd, hidden)
                
            self.sig_slctd = 1 / (1 + np.exp(-node_slctd * n_sign))

            d_n_s =  alpha * self.sig_slctd * (1 - self.sig_slctd) * n_sign
                
            d_hidden = np.dot(d_n_s.reshape(1, -1), W_slctd).reshape(-1)
            d_W_s = np.dot(d_n_s.reshape(-1, 1), hidden.reshape(1, -1))
                
            ### hidden   : (H, )
            ### W_slctd  : (N+1, H)
            ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

            self.params[1][tot_id] -= d_W_s
            
        #d_hidden /= len(contexts)
        self.params[0][target] -= d_hidden

        return None

    def loss(self, target=None):
        if self.HS:
            return -sum(np.log(self.sig_slctd))
        else:
            return np.sum(np.log(self.sig_slctd + 1e-7))


class SkipGram_B2:
    def __init__(self, HS, hidden_size, Data, Data_sub = 5):
        '''HS = True 인 경우, Data에 codebook / sub에 nodebook
                False인 경우, Data에 NS_Freq  / sub에 ns_size
        '''
        self.HS = HS

        if self.HS:
            self.codebook = Data
            self.nodebook = Data_sub
            V = len(self.codebook)
        else:
            self.NS_freq = Data
            self.num_samples = Data_sub
            V = self.NS_freq[-1]+1

            #self.sign =  - np.ones(self.num_samples+1, int)
            self.sign = np.zeros(self.num_samples+1, int)
            self.sign[0] = 1
            
        
        self.H = hidden_size

        # 가중치 초기화
        W_in = 1.42/np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        if HS:
            W_out = 1.42/np.sqrt(V-1 + self.H) * np.random.randn(V-1, self.H).astype('f')
        else:
            W_out = 1.42/np.sqrt(V + self.H) * np.random.randn(V, self.H).astype('f')
        
        self.params = [W_in, W_out]
        

        self.word_vecs = self.params[0]
        self.sig_slctd = None
        #self.time = []
    
    def learn(self, contexts, target, alpha):
        """function 'learn' does not return loss.
            if you want to see it, call 'loss'(target) after learn()
        """
        #W_in, W_out = self.params[0], self.params[0]
        #d_W_in, d_W_out = self.grads[0], self.grads[1]
        hidden = self.params[0][contexts].reshape(-1, self.H)
        
        if self.HS: #### HS
            W_slctd = self.params[1][self.nodebook[target]]

            node_slctd = np.dot(hidden, W_slctd.T)
            self.sig_slctd = 1 / (1 + np.exp(-node_slctd))
                
            #### BackProp
            d_n_s = alpha * (self.sig_slctd + self.codebook[target] - 1)
            
            d_hidden = np.dot(d_n_s, W_slctd)
            
            d_W_s = np.dot(d_n_s.T, hidden)
        ### hidden   : (H, )
        ### W_slctd  : (N, H)
        ### d_n_s    : (N, ) N은 node 수, H는 hidden dimension
            self.params[1][self.nodebook[target]] -= d_W_s
            
            
        else: #### NS
            neg_id = self.NS_freq[np.random.randint(0, len(self.NS_freq), self.num_samples)]
            tot_id = np.append([target], neg_id)

            W_slctd = self.params[1][tot_id]
            
            node_slctd = np.dot(hidden, W_slctd.T)
            #node_slctd *= self.sign
            
            self.sig_slctd = 1 / (1 + np.exp(-node_slctd))

            #d_n_s =  alpha * (self.sig_slctd - 1) * self.sign
            #d_n_s =  alpha * self.sig_slctd * (1 - self.sig_slctd) * self.sign
            d_n_s =  alpha * (self.sig_slctd - self.sign) / self.num_samples
            
            d_hidden = np.dot(d_n_s, W_slctd)
            d_W_s = np.dot(d_n_s.T, hidden)
            
            ### hidden   : (H, )
            ### W_slctd  : (N+1, H)
            ### d_n_s    : (N+1, ) N은 N_samples + 1 , H는 hidden dimension

            self.params[1][tot_id] -= d_W_s
            
        #d_hidden /= len(contexts)
        self.params[0][contexts] -= d_hidden.squeeze()

        return None

    def loss(self, target = None):
        """if teh model is built as Neg_sampling,
            you don't have to input 'target'."""
        if self.HS:
            prop_slctd = np.array(list(map(lambda x, y: 1-x if y else x, self.sig_slctd, self.codebook[target])))
            return -np.sum(np.log(prop_slctd + 1e-7))
        else:
            #return np.sum(np.log(self.sig_slctd + 1e+7))
            prop_slctd = np.array(list(map(lambda x, y: x if y else 1-x, self.sig_slctd, self.sign)))
            return -np.sum(np.log(prop_slctd + 1e-7))