import torch
import torch.nn as nn
from collections import OrderedDict
import math

from src.Func import init_weight

class ResidualNNorm(nn.Module):
    """
    It is for the Residual Connection.
    You can define "res" manually, if it is needed.
    """
    def __init__(self, Layer:nn.Module, dim, dropout=.1, eps=1e-6, res = lambda x:x, mix = False):
        super(ResidualNNorm, self).__init__()
        self.Layer = Layer
        self.res = res
        self.mix = mix

        self.Drop = nn.Dropout(dropout)
        self.Norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, y=None, mask=None):
        """
        if mix=True, Layer gets (x,y), and "res" gets x only.
        """
        if self.mix:
            y = self.Drop(self.Layer(self.Norm(x), y, mask=mask))
        else:
            y = self.Drop(self.Layer(self.Norm(x), mask=mask))
        
        x = self.res(x) + y
        
        return x


class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=.1):
        super(FeedForward, self).__init__()
        
        self.Layers = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(model_dim, hidden_dim, bias=True)),
            ('Relu', nn.ReLU()),
            ('Drop', nn.Dropout(dropout)),
            ('FC2', nn.Linear(hidden_dim, model_dim, bias=True))
        ]))

        init_weight(self.Layers.FC1)
        init_weight(self.Layers.FC2)
    
    def forward(self, x, mask=None):
        # input "mask" exists for format
        return self.Layers(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=200, dropout=.1, scale_embedding=True):
        
        if embedding_dim % 2:
            ValueError("\"Embedding Dimension\" Must be Even-Sized.")
        dim_pos = embedding_dim//2

        super(PositionalEmbedding, self).__init__()
        self.Embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.Dropout = nn.Dropout(dropout)

        self.register_buffer('trigonal', self._make_trigonal_table(dim=dim_pos, max_len=max_len))
        self.vocab_size = vocab_size
        self.scale = scale_embedding
        self.dim_sq = embedding_dim ** .5
        self.max_len = max_len

        init_weight(self.Embedding)

    def forward(self, x):
        if x.size(1) > self.max_len:
            x = x[:, :self.max_len]

        x = self.Embedding(x)
        x *= self.dim_sq
        x += self._get_position(x.size(1))
        x = self.Dropout(x)

        return x

    def generate(self, size, num_samples=10, device='cpu'):
        """For Sentence Gereration"""
        x = [torch.randint(high=self.vocab_size, size=size).to(device) for i in range(num_samples)]
        x = [self.Embedding(i) for i in x]
        x = torch.cat([i.unsqueeze(0) for i in x], dim=0)
        x = torch.mean(x, dim=0)
        if self.scale:
            x *= self.dim_sq
        return x

    def _get_position(self, sent_len):
        return self.trigonal[:sent_len, :]

    def _make_trigonal_table(self, dim, max_len):
        """
        out size : max_len * (2*dim)
        """
        len_size  = torch.arange(max_len).repeat(dim, 1).transpose(1,0) # max_len * dim
        dim_scale = torch.pow(10000, 2*torch.arange(dim)/dim) # dim
        pos_s = torch.sin(len_size / dim_scale)
        pos_c = torch.cos(len_size / dim_scale)
        out = torch.cat((pos_s, pos_c), axis=0) # (2*max_len) * dim
        out = out.reshape(max_len, -1) # max_len * (2*dim)

        d_model = 2*dim
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        pe.requires_grad = False
        return out

class Mask(nn.Module):
    def __init__(self,):
        super(Mask, self).__init__()
    
    def forward(self, x, position=True, pad=0):
        b, l = x.size()

        if position:    # shape : B*1*L*L
            pos_mask = torch.ones(b,l,l).to(x.device)
            pos_mask = pos_mask.tril()
            # B*L*L
            pad_mask = torch.ones(b,l).to(x.device)
            pad_mask[x==0] = 0
            pad_mask = pad_mask.unsqueeze(1)
            # B*L*L
            mask = torch.min(pos_mask, pad_mask).unsqueeze(1)

        else:           # Shape : B*1*1*L
            mask = torch.ones(b,l).to(x.device)
            mask[x==0]=0
            mask = mask.unsqueeze(1).unsqueeze(1)

        return mask
