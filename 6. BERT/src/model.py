import torch
import torch.nn as nn
from collections import OrderedDict
from src.layer import BERT_Block, BERT_Embedding, BERT_Pool

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        
        self.Embedding = BERT_Embedding(config)
        self.Layers = nn.ModuleList([BERT_Block(config) for i in range(config.n_layer)])
        
        self.is_pre_norm = config.is_pre_norm
        if self.is_pre_norm:
            self.Norm = nn.LayerNorm(config.hidden)

    def forward(self, tokens, segments):
        x, mask = self.Embedding(tokens, segments)
        for layer in self.Layers:
            x = layer(x, mask)
        
        if self.is_pre_norm:
            x = self.Norm(x)
        return x
    
class BERT_LM(nn.Module):
    def __init__(self, config):
        super(BERT_LM, self).__init__()
        
        self.BERT = BERT(config)
        
        #self.POOL = BERT_Pool(config)
        self.NSP = nn.Linear(config.hidden, 2)
        self.MLM = nn.Linear(config.hidden, config.vocab_size)
        
    def forward(self, tokens, segments, positions):
        x = self.BERT(tokens, segments)
        
        c = self.NSP(x[:,0])
        w = self.MLM(x).index_select(1,positions).transpose(1,2)
        
        return c, w
    
class BERT_GLUE(nn.Module):
    ### All GLUE tasks are sentence-level classification/regression,
    ### So we only use <Cls> token in actual task
    def __init__(self, model, config):
        super(BERT_GLUE, self).__init__()
        self.BERT = model
        
        # task
        self.Task = nn.Sequential(OrderedDict([
            ('Pooler', nn.Linear(config.hidden, config.hidden)),
            ('Activate', nn.Tanh()),
            #('Dropout', nn.Dropout(config.dropout)),
            ('FC', nn.Linear(config.hidden, config.task_cls))
        ]))
        
        #nn.init.xavier_normal_(self.Task.Pooler.weight, .7)
    
    def forward(self, tokens, segment=None):
        x = self.BERT(tokens, segment)
        c = self.Task(x[:,0])

        return c