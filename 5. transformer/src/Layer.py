import torch
import torch.nn as nn

from src.Attention import MultiHeadAttention
from src.SubLayer import ResidualNNorm, FeedForward

"""
This page consists of Modules:
    Encoder, Decoder
    and their direct submodules.
If you're looking for Self-Attention, go to Attention.py
If you're looking for Residual Connection, Position-Wise Feed-Forward or Positional Embedding, go to SubLayer.py
"""


class EncoderCell(nn.Module):
    def __init__(self, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1):
        super(EncoderCell, self).__init__()
        
        self.Attention = ResidualNNorm(
            MultiHeadAttention(num_head=num_head, in_dim=model_dim, value_dim=value_dim, key_dim=key_dim, out_dim=model_dim),
            dim = model_dim, dropout=dropout
        )

        self.FF = ResidualNNorm(
            FeedForward(model_dim=model_dim, hidden_dim=feed_dim, dropout=dropout),
            dim = model_dim, dropout=dropout
        )

    def forward(self, x, mask):
        x = self.Attention(x, mask=mask)
        x = self.FF(x)
        return x

class DecoderCell(nn.Module):
    def __init__(self, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1):
        super(DecoderCell, self).__init__()

        self.SelfAttention = ResidualNNorm(
            MultiHeadAttention(num_head=num_head, in_dim=model_dim, value_dim=value_dim, key_dim=key_dim, out_dim=model_dim),
            dim = model_dim, dropout=dropout
        )
        self.MixAttention = ResidualNNorm(
            MultiHeadAttention(num_head=num_head, in_dim=model_dim, value_dim=value_dim, key_dim=key_dim, out_dim=model_dim, mix=True,),
            dim = model_dim, dropout=dropout, mix=True
        )
        self.FF = ResidualNNorm(
            FeedForward(model_dim=model_dim, hidden_dim=feed_dim, dropout=dropout),
            dim = model_dim, dropout=dropout
        )

    def forward(self, x, enc, mask_x, mask_enc):
        x = self.SelfAttention(x=x, mask=mask_x)
        x = self.MixAttention(x=x, y=enc, mask=mask_enc)
        x = self.FF(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1, return_each = False, eps=1e-6):
        super(Encoder, self).__init__()

        self.Cells = nn.ModuleList([
            EncoderCell(num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout)
            for i in range(num_layers)
        ])

        self.Norm = nn.LayerNorm(model_dim, eps=eps)

        self.return_each = return_each
    
    def forward(self, x, mask):
        if self.return_each:
            out = []
            for cell in self.Cells:
                x = self.Norm(cell(x, mask))
                out.append(x)
            return out
        else:
            for cell in self.Cells:
                x = cell(x, mask)
            return self.Norm(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1, get_each=False, eps=1e-6):
        super(Decoder, self).__init__()

        self.Cells = nn.ModuleList([
            DecoderCell(num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout)
            for i in range(num_layers)
        ])

        self.Norm = nn.LayerNorm(model_dim, eps=eps)

        self.get_each = get_each

    def forward(self, x, enc, mask_x=None, mask_enc=None):
        for i, cell in enumerate(self.Cells):
            x = cell(x, enc=enc[i] if self.get_each else enc, mask_x=mask_x, mask_enc=mask_enc)
        return self.Norm(x)