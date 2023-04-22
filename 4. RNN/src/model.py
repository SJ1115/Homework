import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layer import Encoder, Decoder_Rare, Decoder_with_Attention
from src.func import prop_mode, last_hidden

class Baseline(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, hidden_size, depth, dropout, reverse=True):
        super(Baseline, self).__init__()

        self.Encoder = Encoder(vocab_size=enc_vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, depth=depth, dropout=dropout)

        self.Decoder = Decoder_Rare(vocab_size=dec_vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, depth=depth, dropout=dropout, input_feed=False)

        self.FC = nn.Linear(in_features=hidden_size, out_features=dec_vocab_size, bias=False)

        # Initialization
        self.Encoder.init_weight()
        self.Decoder.init_weight()
        nn.init.uniform_(self.FC.weight, -.1, .1)

    def forward(self, en_sent, de_sent):
        B, L = de_sent.size()
        ### Enc ###
        enc_out, hs = self.Encoder(en_sent)
        #hs = last_hidden(hs)

        ### Dec ###
        # pass Dec at once
        dec_out, _ = self.Decoder(sent=de_sent, feed=None, lstm_hs=hs, train=True)

        ## (B,L,H) -> (B,L,V)
        model_outs = self.FC(dec_out)

        ## (B,L,V) -> (B,V,L)
        model_outs = model_outs.transpose(1,2)
        return model_outs

    def predict(self, en_sent, max_len=51, device='cpu'):
        ### Enc ###
        enc_out, hs = self.Encoder(en_sent)

        ### Dec ###
        # pass Dec one by one
        model_out = []
        
        size = (enc_out.size(0), 1) # (Batch)
        word = torch.ones(size, dtype=torch.int).to(device) # 1 = <S> token

        for i in range(max_len):
            dout, hs = self.Decoder(sent=word, lstm_hs=hs, train=False, word_ind=i)
            
            word = self.FC(dout).transpose(1,2) # B*V*1
            model_out.append(word)
            word = word.argmax(axis=1)

            if size[0] == 1:
                if int(word)==2:
                    break
            else:
                if not int(torch.sum((word != 0) * (word != 2))):
                    break

        # vector form(for PPL check)
        model_out = torch.cat(model_out, axis=2)
        # token form
        words_out = torch.argmax(model_out, axis=1)
        
        return words_out, model_out

###############################################

class NMT(nn.Module):
    def __init__(self, attention, enc_vocab_size, dec_vocab_size, embedding_dim, hidden_size, depth, dropout, input_feed=False, reverse=True, window_size=10, length=52):
        super(NMT, self).__init__()

        self.Encoder = Encoder(vocab_size=enc_vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, depth=depth, dropout=dropout)

        align, score = attention

        self.Decoder = Decoder_with_Attention(align=align, score=score, vocab_size=dec_vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, depth=depth, dropout=dropout, input_feed=input_feed, reverse=reverse, window_size=window_size, length=length)

        self.FC = nn.Linear(in_features=hidden_size*2, out_features=dec_vocab_size, bias=False)

        # Flag for using input feeding 
        self.inptFd = input_feed

        # Initialization
        self.Encoder.init_weight()
        self.Decoder.init_weight()
        nn.init.uniform_(self.FC.weight, -.1, .1)

    def forward(self, en_sent, de_sent):
        ### Enc ###
        enc_out, hs = self.Encoder(en_sent)
        en_len = (en_sent>0).sum(axis=1)#.tolist()
        #hs = last_hidden(hs)
        ### Dec ###
        if self.inptFd:
            # pass Dec 1 by 1
            leng = int((de_sent>0).sum(axis=1).max())
            dec_out = []
            att = enc_out[:,-1,:].unsqueeze(1)

            for i in range(leng):
                out, hs, att = self.Decoder(enc=(enc_out, en_len), dec_sent=de_sent, feed=att, lstm_hs=hs, train=True, word_ind=i)
                dec_out.append(out)
            
            dec_out = torch.cat([tensor for tensor in dec_out], dim=1)
            self.Decoder.reset_buff()
            
        else:
            # pass Dec at once
            dec_out, _, _ = self.Decoder(enc=(enc_out, en_len), dec_sent=de_sent, feed=None, lstm_hs=hs, train=True, word_ind=-1)
        
        ## (B,L,H) -> (B,L,V)
        model_outs = self.FC(dec_out) 
        ## (B,L,V) -> (B,V,L)
        model_outs = model_outs.transpose(1,2) 
        return model_outs

    def predict(self, en_sent, max_len=50, device='cpu'):
        ### Enc ###
        enc_out, hs = self.Encoder(en_sent)
        en_len = (en_sent>0).sum(axis=1)

        ### Dec ###
        # pass Dec one by one
        model_out = []
        att = enc_out[:,-1,:].unsqueeze(1)

        size = (enc_out.size(0),1) # (Batch)
        word = torch.ones(size, dtype=torch.int).to(device) # 1 = <S> token

        for i in range(max_len):
            dout, hs, att = self.Decoder(enc=(enc_out, en_len), dec_sent=word, feed=att, lstm_hs=hs, train=False, word_ind=i)
            
            word = self.FC(dout)
            model_out.append(word)
            word = word.argmax(axis=2)

            if size[0] == 1:
                if int(word)==2:
                    break
            else:
                if not int(torch.sum((word != 0) * (word != 2))):
                    break

        self.Decoder.reset_buff()

        # vector form(for PPL check)
        model_out = torch.cat([tensor for tensor in model_out], axis=1).transpose(1,2)
        # token form
        words_out = model_out.argmax(axis=1)
        
        return words_out, model_out

## end