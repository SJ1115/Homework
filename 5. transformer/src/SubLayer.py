import torch
import torch.nn as nn

class ResidualNNorm(nn.Module):
    """
    It is for the Residual Connection.
    You can define "res" manually, if it is needed.
    """
    def __init__(self, Layer, dim, dropout=.1, eps=1e-6, res = lambda x:x, mix = False):
        super(ResidualNNorm, self).__init__()
        self.Layer = Layer
        self.res = res
        self.mix = mix

        self.Drop = nn.Dropout(dropout)
        self.Norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, y=None):
        """
        if mix=True, Layer gets (x,y), and "res" gets x only.
        """
        if self.mix:
            x = self.Drop(self.Layer(x, y)) + self.res(x)
        else:
            x = self.Drop(self.Layer(x)) + self.res(x)
        
        x = self.Norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super(FeedForward, self).__init__()
        
        self.FC1 = nn.Linear(model_dim, hidden_dim, bias=True)
        self.FC2 = nn.Linear(hidden_dim, model_dim, bias=True)
        self.Relu = nn.ReLU()
    
    def forward(self, x):
        x = self.FC1(x)
        x = self.Relu(x)
        x = self.FC2(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=200, dropout=.1, scale_embedding=True):
        
        if embedding_dim % 2:
            ValueError("\"Embedding Dimension\" Must be Even-Sized.")
        dim_pos = embedding_dim//2

        super(PositionalEmbedding, self).__init__()
        self.Embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.Dropout = nn.Dropout(dropout)

        self.register_buffer('trigonal', self._make_trigonal_table(dim=dim_pos, max_len=max_len))
        self.vocab_size = vocab_size
        self.scale = scale_embedding
        self.dim_sq = embedding_dim ** .5

    def forward(self, x):
        x = self.Embedding(x) + self._get_position(x.size(1))
        x = self.Dropout(x)
        if self.scale:
            x *= self.dim_sq
        return x

    def generate(self, size, num_samples=10):
        """For Sentence Gereration"""
        x = [torch.randint(high=self.vocab_size, size=size) for i in range(num_samples)]
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
        dim_scale = torch.pow(10000, torch.arange(dim)/dim) # dim
        pos_s = torch.sin(len_size / dim_scale)
        pos_c = torch.cos(len_size / dim_scale)
        out = torch.cat((pos_s, pos_c), axis=0) # (2*max_len) * dim
        out = out.reshape(max_len, -1)
        return out