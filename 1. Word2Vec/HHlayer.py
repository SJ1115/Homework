import sys
sys.path.insert(0, "./DL_Scratch_RFR")


from DL_Scratch_RFR.common.np import np


class Sep_nodes:
    def __init__(self, huffman):
        self.params, self.grads = [], []
        self.nodes = None
        self.huffman = huffman
        
        self.value = None
        
    def forward(self, h, t):
        self.value = np.zeros_like(h)
        if not isinstance(t, (list, np.ndarray)):
            t = [t]
            h = np.array(h)
        
        self.nodes = [np.array(self.huffman.nodebook[t_]) for t_ in t]
        print(h.shape)
        return [h[i, nodes] for i, nodes in enumerate(self.nodes)]
    
    def backward(self, dout):
        for i, do in enumerate(dout):
            self.value[i,self.nodes[i]] += do
        
        return self.value
        
class Sigmoid_sep:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        self.flag = None

    def forward(self, x):
        self.flag = isinstance(x, list)
        if self.flag:
            out = [1 / (1 + np.exp(-x_)) for x_ in x]
            self.out = out
        else:
            out = 1 / (1 + np.exp(-x))
            self.out = out
        return out

    def backward(self, dout):
        if self.flag:
            dx = [do * (1.0 - o) * o for do, o in zip(dout, self.out)]
        else:
            dx = dout * (1.0 - self.out) * self.out
        return dx

class HHSLoss():
    ## 여기서는 Softmzx부분만 우선 만들어보기
    def __init__(self, huffman):
        self.params, self.grads = [], []
        self.y = None  # tree로 내려가는 확률값
        self.t = None  # 정답 레이블
        self.flag = None

        self.huffman = huffman

        self.params, self.grads = [], []


    def forward(self, h, t):
        """Huffman Tree Based Hierachical Softmax
        Input : log_2_V 차원 array x, true word t
        Output : Softmax를 통과한 것과 같은 결과

        """
        self.t = t
        self.y = h

        self.flag = isinstance(t, (list, np.ndarray)) 

        if not self.flag:
            self.y = [np.log(1-j + 1e-7) if i else np.log(j)
                          for i, j in zip(self.huffman.codebook[t], h)]
            loss = np.sum(self.y) 

        else:
            self.y = [
                        [np.log(1-j + 1e-7) if i else np.log(j)
                             for i, j in zip(self.huffman.codebook[t_], h_)]
                                for h_, t_ in zip(h, t)]
            loss = [ np.sum(y_) for y_ in self.y]

        return -np.mean(loss)

    def backward(self, dout=1):
        if self.flag:
            batch_size = len(self.y)

            dy = [-np.reciprocal(y) / batch_size for y in self.y]
            dx = [-y* dout * (2*np.array(self.huffman.codebook[t])-1) for y, t in zip(dy, self.t)]
        else:
            dy = -np.reciprocal(self.y)
            dx = -dy * dout * (2*np.array(self.huffman.codebook[self.t])-1)
        return dx

class Dot_sep:
    def __init__(self, W, huffman):
        self.huffman = huffman
        self.params = [W]; self.grads = [np.zeros_like(W)]

        self.h = None
        self.t_nodes = None
        self.W_sep = None


    def forward(self, h, t):
        W, = self.params
        self.h = h

        self.t_nodes = [self.huffman.nodebook[t_] for t_ in t]
        self.W_sep = [W[t_node] for t_node in self.t_nodes]

        Out_sep = [np.dot(W_part, h[i]) for i, W_part in enumerate(self.W_sep)]

        return Out_sep

    def backward(self, dout):
        dW = np.zeros_like(self.params[0])

        d_h_s = [np.dot(W_s.T, do_s.reshape(-1,1)) 
                     for W_s, do_s in zip(self.W_sep, dout)]

        d_W_s = [np.dot(do_s.reshape(-1,1), h_s.reshape(1,-1))
                     for h_s, do_s in zip(self.h, dout)]

        np.add.at(dW,
                  [node for t_node in self.t_nodes for node in t_node],
                  [d_w for d_W in d_W_s for d_w in d_W])
        self.grads[0][...] = dW
        d_h_s = np.hstack(d_h_s).T
        return d_h_s

class HierarchicalSoftmaxLoss:
    def __init__(self, W, huffman):
        self.layers = [
            Dot_sep(W, huffman),
            Sigmoid_sep(),
            HHSLoss(huffman)
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        ## Dot_seperating
        h = self.layers[0].forward(h, target)
        ## Sigmoid_seperated
        h = self.layers[1].forward(h)
        ## Hierarchical_Softmax
        loss = self.layers[2].forward(h, target)
        return loss

    def backward(self, dout = 1):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
  