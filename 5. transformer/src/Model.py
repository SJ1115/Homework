import torch.nn as nn

from src.Layer import Encoder, Decoder
from src.SubLayer import PositionalEmbedding

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1, send_each=False, embed_share=True, scale_embedding=True):
        super(Transformer, self).__init__()

        self.EmbeddingEnc = PositionalEmbedding(vocab_size=vocab_size, embedding_dim=model_dim, max_len=max_len, dropout=dropout, scale_embedding=scale_embedding)
        self.EmbeddingDec = PositionalEmbedding(vocab_size=vocab_size, embedding_dim=model_dim, max_len=max_len, dropout=dropout, scale_embedding=scale_embedding)
        
        self.Encoder = Encoder(num_layers=num_layers, num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout, return_each=send_each)
        self.Decoder = Decoder(num_layers=num_layers, num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout, get_each=send_each)
        
        self.OutFC = nn.Linear(model_dim, vocab_size, bias=False)

        if embed_share:
            self.EmbeddingDec.Embedding.weight = self.EmbeddingEnc.Embedding.weight
            self.OutFC.weight = self.EmbeddingDec.Embedding.weight
        self.max_len = max_len

    def forward(self, src, trg):
        src = self.EmbeddingEnc(src)
        src = self.Encoder(src)
        
        trg = self.EmbeddingDec(trg)
        trg = self.Decoder(trg, src)
        
        trg = self.OutFC(trg)
        return trg

    def predict(self, src, max_len = None, num_samples=10):
        max_len = self.max_len if not max_len else max_len
        
        src = self.EmbeddingEnc(src)
        src = self.Encoder(src)

        size = (src.size(0), max_len) # Batch * Len

        trg = self.EmbeddingDec.generate(size, num_samples=num_samples)
        trg = self.Decoder(trg, src)
        
        trg = self.OutFC(trg) # Batch*
        return trg.argmax(dim=-1)