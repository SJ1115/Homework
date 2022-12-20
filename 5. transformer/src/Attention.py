import torch
import torch.nn as nn

# Scaled Dot Product Attention
class SelfAttention(nn.Module):
    def __init__(self, dim, mask = False, dropout_rate = .1):
        """dim is the dimension of Query&Key, NOT Value"""
        super(SelfAttention, self).__init__()
        self.mask = mask
        self.scaler = 1 / torch.sqrt(torch.tensor(dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        """
        notation in einsum:
        b : Batch       l(m) : Len of sentence, 'm' is used when different length exists.
        d : Dim(K&Q)    c : Channel size(# of head)
        h : Hidden = dim(V)
        """
        score = torch.einsum("blcd,bmcd->bclm", query, key) * self.scaler
        if self.mask:
            masking = torch.ones(query.shape[1], key.shape[1]).tril(diagonal=-1)
            score = score.masked_fill(masking==0, -1e+7)
        score = self.dropout(torch.softmax(score, axis=3))
        out = torch.einsum("bmch,bclm->blch", value, score)
        
        return out
        
        """ This was a proto-type. keep now
        score = torch.einsum("blcd,bmcd->blmc", query, key) * self.scaler
        if self.mask:
            masking = torch.ones(query.shape[1], key.shape[1]).tril()
            score = torch.einsum("blmc,lm->blmc", score, masking)
        score = self.dropout(torch.softmax(score, axis=1))
        out = torch.einsum("bmch,blmc->blch", value, score)
        """
        
        
# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, in_dim, value_dim, key_dim, out_dim, mix=False, mask=False, dropout=.1):

        super(MultiHeadAttention, self).__init__()

        self.W_q = nn.Linear(in_dim, num_head * key_dim, bias=False)
        self.W_k = nn.Linear(in_dim, num_head * key_dim, bias=False)
        self.W_v = nn.Linear(in_dim, num_head * value_dim, bias=False)
        self.W_out = nn.Linear(num_head * value_dim, out_dim, bias=False)

        self.attention = SelfAttention(dim=key_dim, mask=mask, dropout_rate=dropout)

        self.n_head = num_head
        self.mix = mix

    def forward(self, x, y=None):
        query = self.W_q(x)
        key   = self.W_k(y if self.mix else x)
        value = self.W_v(y if self.mix else x)

        b, l, _ = query.shape
        _, m, _ = key.shape
        
        query = query.view(b, l, self.n_head, -1)
        key   = key.view(b, m, self.n_head, -1)
        value = value.view(b, m, self.n_head, -1)

        out = self.attention(query, key, value).view(b, l, -1)
        out = self.W_out(out)

        return out
