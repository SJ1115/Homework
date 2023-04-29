import torch
import torch.nn as nn

from src.Layer import Encoder, Decoder
from src.SubLayer import PositionalEmbedding, Mask
from src.Func import init_weight

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, num_head, model_dim, value_dim, key_dim, feed_dim, dropout=.1, send_each=False, embed_share=True, scale_embedding=True):
        super(Transformer, self).__init__()

        self.Mask = Mask()

        self.EmbeddingEnc = PositionalEmbedding(vocab_size=vocab_size, embedding_dim=model_dim, max_len=max_len, dropout=dropout, scale_embedding=scale_embedding)
        self.EmbeddingDec = PositionalEmbedding(vocab_size=vocab_size, embedding_dim=model_dim, max_len=max_len, dropout=dropout, scale_embedding=scale_embedding)
        
        self.Encoder = Encoder(num_layers=num_layers, num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout, return_each=send_each)
        self.Decoder = Decoder(num_layers=num_layers, num_head=num_head, model_dim=model_dim, value_dim=value_dim, key_dim=key_dim, feed_dim=feed_dim, dropout=dropout, get_each=send_each)
        
        self.OutFC = nn.Linear(model_dim, vocab_size, bias=False)

        if embed_share:
            self.EmbeddingDec.Embedding.weight = self.EmbeddingEnc.Embedding.weight
            self.OutFC.weight = self.EmbeddingDec.Embedding.weight
        else:
            # Embedding/Encoder/Decoder initialize themselves.
            init_weight(self.OutFC)
        
        self.max_len = max_len 
        self.model_dim = model_dim # used in Trainer.py

    def _encode(self, src):
        src_mask = self.Mask(src, position=False)
        src = self.EmbeddingEnc(src)
        src = self.Encoder(src, src_mask)
        return src, src_mask
        
    def _decode(self, src, src_mask, trg):
        trg_mask = self.Mask(trg, position=True)
        trg = self.EmbeddingDec(trg)
        trg = self.Decoder(trg, src, trg_mask, src_mask)        
        out = self.OutFC(trg)
        return out

    def forward(self, src, trg):
        src, src_mask = self._encode(src)
        
        out = self._decode(src, src_mask, trg)
        
        return out.transpose(1,2)
        
    def predict(self, src, beam_size = 4, alpha=.6, max_len = None, device='cpu'):
        """
        predict with beam search, and it does not support batch calculation.
        only B = 1 size is available.
        """
        if src.size(0) > 1:
            ValueError("Batch input must be 1")

        V = self.EmbeddingDec.Embedding.weight.size(0)
        B = beam_size
        a = alpha
            
        max_len = self.max_len if not max_len else max_len
        
        src, src_mask = self._encode(src)

        out = torch.ones((1,1), dtype=torch.int).to(device) # 1 = <Sos>
        score = torch.zeros((1,)).to(device)
        
        for i in range(max_len):
            dec_out = self._decode(src, src_mask, out)[:, -1, :]

            # Beam Search
            prob = dec_out.log_softmax(dim=-1)
            log_prob = prob / ((i+1)/6)**a
            
            eos_flag = out[:,-1] == 2
            log_prob[eos_flag, :] = 0
            
            score = score.unsqueeze(1) + log_prob
            score, indices = score.reshape(-1).topk(B)
            
            beam_ids = torch.divide(indices, V, rounding_mode='floor')
            token_ids = torch.remainder(indices, V)

            next_in = []
            for beam_i, token_i in zip(beam_ids, token_ids):
                prev_in = out[beam_i]
                if prev_in[-1] == 2:
                    token_i = torch.tensor(0).to(device) # <Pad> after <EOS>
                token_i = token_i.unsqueeze(0)
                next_in.append(torch.cat([prev_in, token_i]))
            out = torch.vstack(next_in)
            
            if not int(torch.sum((out!=0) * (out!=2))):
                break
            
            if not i:
                src = src.repeat_interleave(B, 0)
                src_mask = src_mask.repeat_interleave(B, 0)
        
        out, _ = max(zip(out, score), key=lambda x: x[1])
        out = out.unsqueeze(0)
        return out
    
    def New_predict(self, src, max_len = None, device='cpu'):
        """
        Predict with Simple, Greedy Search. it is fast and bach-applicable.
        """
        max_len = self.max_len if not max_len else max_len
        
        src, src_mask = self._encode(src)

        size = (src.size(0), 1) # Batch * Len
        trg = torch.ones(size, dtype=torch.int).to(device) # 1 = <Sos>

        for i in range(max_len):
            out = self._decode(src, src_mask, trg)[:,-1,:].argmax(dim = -1)
            
            trg = torch.cat((trg, out.unsqueeze(1)), axis=1)
            
            if size[0] == 1:
                if int(out)==2: # 2 = <EOS>, 0 = <Pad>
                    break
            else:
                if not int(torch.sum((out != 0) * (out != 2))):
                    break
        return trg

# End