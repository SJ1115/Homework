import torch
import torch.nn as nn
from src.Func import init_weight

# Scaled Dot Product Attention
class SelfAttention(nn.Module):
    def __init__(self, dim, dropout_rate = .1):
        """dim is the dimension of Query&Key, NOT Value"""
        super(SelfAttention, self).__init__()
        self.scaler = dim ** -.5
        self.dropout = nn.Dropout(dropout_rate)
        self.__attention__ = None

    def forward(self, query, key, value, mask=None):
        """
        notation in einsum:
        b : Batch       l(m) : Len of sentence, 'm' is used when different length exists.
        d : Dim(K&Q)    c : Channel size(# of head)
        h : Hidden = dim(V)
        """
        score = torch.matmul(query, key.transpose(2,3)) * self.scaler
        # BCLD * BCDM(from BCMD) -> BCLM
        if mask != None:
            score = score.masked_fill(mask==0, -1e+9)
        
        score = torch.softmax(score, axis=3)
        # softmax to M, refer'ee'.
        self.__attention__ = self.dropout(score)## keep for visualization

        out = torch.matmul(score, value)
        # BCLM * BCMH -> BCLH
        
        return out.transpose(1,2).contiguous() ## BLCH

        """# This was a proto-type. keep now
        score = torch.einsum("blcd,bmcd->blmc", query, key) * self.scaler
        if self.mask:
            masking = torch.ones(query.shape[1], key.shape[1]).tril()
            score = torch.einsum("blmc,lm->blmc", score, masking)
        score = self.dropout(torch.softmax(score, axis=1))
        out = torch.einsum("bmch,blmc->blch", value, score)
        """
        
        
# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, in_dim, value_dim, key_dim, out_dim, mix=False, dropout=.1):

        super(MultiHeadAttention, self).__init__()

        self.W_q   = nn.Linear(in_dim, num_head * key_dim, bias=True)
        self.W_k   = nn.Linear(in_dim, num_head * key_dim, bias=True)
        self.W_v   = nn.Linear(in_dim, num_head * value_dim, bias=True)
        self.W_out = nn.Linear(num_head * value_dim, out_dim, bias=True)

        self.attention = SelfAttention(dim=key_dim, dropout_rate=dropout)

        self.n_head = num_head
        self.mix = mix

        self.init_weight()

    def forward(self, x, y=None, mask=None):
        query = self.W_q(x)
        if self.mix:
            key   = self.W_k(y)
            value = self.W_v(y)
        else:
            key   = self.W_k(x)
            value = self.W_v(x)
            
        b, l, _ = query.shape
        _, m, _ = key.shape
        
        query = query.view(b, l, self.n_head, -1).transpose(1,2) # B*C*L*D
        key   = key.view(b, m, self.n_head, -1).transpose(1,2)   # B*C*M*D
        value = value.view(b, m, self.n_head, -1).transpose(1,2) # B*C*M*H

        out = self.attention(query, key, value, mask=mask).view(b, l, -1)
        out = self.W_out(out)

        return out
    
    def init_weight(self,):
        init_weight(self.W_q, True)
        init_weight(self.W_k, True)
        init_weight(self.W_v, True)
        init_weight(self.W_out, True)
