import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors

####
def init_module(param, init=None):
    if init == 'he':
        return nn.init.kaiming_uniform_(param, nonlinearity='relu')
    elif init == 'xavier':
        return nn.init.xavier_uniform_(param)
    elif init == 'base':
        return nn.init.normal_(param, 0, .01)
    else:
        return

class CNN_TC(nn.Module):
    def __init__(self, method, data, wv, kernels = [3,4,5], feature = 100, dropout_ratio = .5, init = ('he', 'xavier'), bias = True, device = None):
        super(CNN_TC, self).__init__()
        
        self.device = device
        
        vocab_size = len(data['w2i'])
        label_size = max(data['train_label'])+1

        self.sent_len = data['train'].shape[1]
        self.feature = feature
        
        ## Embedding
        self.method = method.lower()
        if self.method not in ('rand', 'static', 'nonstatic', 'multichannel'):
            raise ValueError("Parameter 'method' not proper.")
        
        if self.method == 'rand':
            self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0, dtype=torch.float)
        elif method in ('static', 'nonstatic'):
            freeze = (method == 'static')
            self.embedding = nn.Embedding.from_pretrained(wv, padding_idx=0, freeze = freeze)
        else:
            self.embedding = nn.ModuleList([ 
                nn.Embedding.from_pretrained(wv, padding_idx=0, freeze = False),
                nn.Embedding.from_pretrained(wv, padding_idx=0, freeze = True)
            ])
            
        channel_out = (self.method == "multichannel") + 1
        
        ### Convolution ###
        self.kernels = list(kernels)
        #self.convnet_list = nn.ModuleList([nn.Conv1d(300, feature, k, dtype=torch.float, device = self.device) for k in self.kernels])
        self.convnet_list = nn.ModuleList([nn.Conv1d(1, self.feature, 300*k, stride=300,dtype=torch.float, device = self.device) for k in self.kernels])
        
        ### FC & Dropout
        self.FC = nn.Linear(channel_out * len(self.kernels) * self.feature, label_size, bias=bias, dtype=torch.float)
        #### Dropout will be declared in forward(), by nn.F
        self.dropout_ratio =dropout_ratio
        #### We don't implement Softmax, since Loss func already included it. 
        
        #Initialize
        if self.method =='rand':
            nn.init.uniform_(self.embedding.weight, -.25, .25)
        for Conv in self.convnet_list:
            init_module(Conv.weight, init[0])
        init_module(self.FC.weight, init[1])

    def forward(self, x):
        if self.method == 'multichannel':
            embed_x = [embed(x).view(-1, 1, 300*self.sent_len) for embed in self.embedding]
#            embed_x = [x.view(len(x), 1, -1) for x in embed_x]

            #conv_out = [sum([conv(x) for x in embed_x]) for conv in self.convnet_list]
            conv_out = [F.relu(conv(x)) for x in embed_x for conv in self.convnet_list]
            pool_out = [F.max_pool1d(out, self.sent_len - k + 1).view(-1, self.feature) for (out, k) in zip(conv_out, self.kernels*2)]
        
        else:
            x = self.embedding(x)
            x = x.view(-1, 1, 300*self.sent_len)
            conv_out = [F.relu(conv(x)) for conv in self.convnet_list ]
        
            pool_out = [F.max_pool1d(out, self.sent_len - k + 1).view(-1, self.feature) for out, k in zip(conv_out, self.kernels)]

            """pool_out = []
            for i, (conv, k) in enumerate(zip(self.convnet_list, self.kernels)):
                t = conv(x)
                t = F.relu(t)
                t = F.max_pool1d(t, t.shape[2]-k+1)
                pool_out.append(t)"""
        x = torch.cat(pool_out, 1)
        
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.FC(x)
        
        #x = F.softmax(x, dim=1)
        
        return x
    

###########RE-TRIAL

class CNN_New(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_New, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"] ## 100
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        #assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "nonstatic" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(self.WV_MATRIX)
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(self.WV_MATRIX)
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(self.FILTER_NUM*len(self.FILTERS), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        print(x.shape)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM)
            for i in range(len(self.FILTERS))]
        print([c.shape for c in conv_results])
        x = torch.cat(conv_results, 1)
        print(x.shape)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x
 