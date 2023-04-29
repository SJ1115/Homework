import torch
import torch.nn as nn
from collections import OrderedDict

######## Low Level Gadgets #######
# 1. SelfAttention
# 2. FeedForward
# 3. Mask builder

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        
        assert ((config.hidden % config.n_head) ==0)
        self.n_head = config.n_head
        
        self.FC = nn.ModuleDict({
            "Query": nn.Linear(config.hidden, config.hidden),
            "Key"  : nn.Linear(config.hidden, config.hidden),
            "Value": nn.Linear(config.hidden, config.hidden),
            "Out"  : nn.Linear(config.hidden, config.hidden)
        })
        
        self.Drop_in  = nn.Dropout(config.dropout)
        self.Drop_out = nn.Dropout(config.dropout)
        self.Norm = nn.LayerNorm(config.hidden, eps=config.norm_eps)
        
        self.scaler = config.hidden ** -.5
        self.is_pre_norm = config.is_pre_norm
        ## initialize
        for fc in self.FC.values():
            nn.init.xavier_normal_(fc.weight, gain=.65)

    def forward(self, x, mask=None):
        if self.is_pre_norm:
            out = self.Norm(x)
        else:
            out = x

        out = self._self_attention(out, mask)
        out = self.Drop_out(out)

        if self.is_pre_norm:
            out += x
        else:
            out = self.Norm(out + x)
        return out

    def _self_attention(self, x, mask=None):
        b, l, _ = x.shape
        query = self.FC.Query(x).view(b, l, self.n_head, -1).transpose(1,2)
        key   = self.FC.Key(x).view(b, l, self.n_head, -1).transpose(1,2)
        value = self.FC.Value(x).view(b, l, self.n_head, -1).transpose(1,2)
        
        score = torch.matmul(query, key.transpose(2,3)) * self.scaler
        if mask != None:
            score = score.masked_fill(mask==0, -1e+9)
            
        score = self.Drop_in(torch.softmax(score, axis=3))
        out = torch.matmul(score, value).transpose(1,2).contiguous().view(b,l,-1)
        out = self.FC.Out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        
        self.is_pre_norm = config.is_pre_norm

        self.Layers = nn.Sequential(OrderedDict([
            ('FC_in', nn.Linear(config.hidden, config.FF_hidden)),
            ('Activate', nn.GELU()),
            ('Drop', nn.Dropout(config.dropout)),
            ('FC_out', nn.Linear(config.FF_hidden, config.hidden))
        ]))
        
        #if self.is_pre_norm:
        #    del self.Layers.Drop
        
        self.Drop = nn.Dropout(config.dropout)
        self.Norm = nn.LayerNorm(config.hidden, eps=config.norm_eps)

        
    def forward(self, x):
        if self.is_pre_norm:
            x = self.Drop(self.Layers(self.Norm(x))) + x
        else:
            x = self.Norm(self.Drop(self.Layers(x)) + x)
        return x

class Mask(nn.Module):
    def __init__(self, config):
        super(Mask, self).__init__()
    
        self.padding_idx = config.padding_idx
        self.masking_idx = config.masking_idx
        
    def forward(self, x):
        b, l = x.size()

        # Shape : B*1*1*L
        mask = torch.ones(b,l).to(x.device)
        mask[x==self.padding_idx]=0
        mask[x==self.masking_idx]=0
        mask = mask.unsqueeze(1).unsqueeze(1)

        return mask

####### Intermediate Level Gadgets #######
# 1. BERT_Embedding
    # contains Mask builder(3)
# 2. BERT_Block
    # contains SelfAttention(1) & FeedForward(2)
# 3. BERT_Pool

class BERT_Embedding(nn.Module):
    def __init__(self, config):
        super(BERT_Embedding, self).__init__()
        
        self.Mask = Mask(config)
        self.WordEmbedding = nn.Embedding(config.vocab_size, config.hidden, padding_idx=0)
        self.PositionEmbedding = nn.Embedding(config.max_len, config.hidden)
        self.SegmentEmbedding = nn.Embedding(config.n_segment, config.hidden)
        
        self.register_buffer("position", torch.arange(config.max_len).expand((1, -1)))
        
        self.is_pre_norm = config.is_pre_norm
        if not self.is_pre_norm:
            self.Norm = nn.LayerNorm(config.hidden)
        self.Drop = nn.Dropout(config.dropout)
        ## initialize
        
    def forward(self, tokens, segments):
        
        mask = self.Mask(tokens)
        
        pos = self.position[:, 0:tokens.size(1)]
        x = self.WordEmbedding(tokens) + \
            self.PositionEmbedding(pos) + \
            self.SegmentEmbedding(segments)
        
        if not self.is_pre_norm:
            x = self.Norm(x)
        
        x = self.Drop(x)
        
        return x, mask
    
class BERT_Block(nn.Module):
    def __init__(self, config):
        super(BERT_Block, self).__init__()
        
        self.SelfAttention = SelfAttention(config)
        self.FeedForward   = FeedForward(config)
        
    def forward(self, x, mask=None):
        return self.FeedForward(self.SelfAttention(x, mask))
    
class BERT_Pool(nn.Module):
    def __init__(self, config):
        super(BERT_Pool, self).__init__()
        
        self.Layers =  nn.Sequential(OrderedDict([
            ("Pooler", nn.Linear(config.hidden, config.hidden)),
            ("Activate", nn.Tanh()),
        ]))
        
    def forward(self, x):
        return self.Layers(x[:,0])
####### Equipable(for Fine-Tuning) Gadgets #######
# 1. 