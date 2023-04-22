import torch
import torch.nn as nn

from src.attention import Attention_Global, Attention_Local_Monotonic, Attention_Local_Predictive
from src.func import align_check, init_lstm

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, depth, dropout):
        super(Encoder, self).__init__()

        self.Embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.Dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=depth, dropout=dropout, batch_first=True)

    def forward(self, sent):
        embed = self.Embedding(sent)
        embed = self.Dropout(embed)

        #embed = nn.utils.rnn.pack_padded_sequence(embed, (sent>0).sum(axis=1).tolist(), enforce_sorted=False, batch_first=True)
        outs, hs = self.LSTM(embed)
        #outs, _ = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True, padding_value=0)
        
        return outs, hs
        # seq_len is used only in local-predictive attention

    def init_weight(self):
        nn.init.uniform_(self.Embedding.weight, -.1, .1)
        init_lstm(self.LSTM)

class Decoder_Rare(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, depth, dropout, input_feed):
        super(Decoder_Rare, self).__init__()

        self.input_feed = input_feed
        if input_feed:
            input_size = embedding_dim+hidden_size
        else:
            input_size = embedding_dim

        self.Embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.Dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=depth, dropout=dropout, batch_first=True)

        self.embed_buff = None

    def forward(self, sent, feed=None, lstm_hs=None, train=True, word_ind=0):
        """
        prop_mode, word_ind is used only in input_feed == True.
            args:
        sent : input sentence, size(B*L), dtype(int)
        feed : additional input for Input Feeding. size(B*H)
        train: True for train(forward() in model),
               False for test(predict() in model).
        """
        # Embedding
        if train:
            if self.input_feed:
                if self.embed_buff == None:
                    self.embed_buff = self.Embedding(sent)
                embed = self.embed_buff[:, word_ind, :].unsqueeze(1)
            else:
                embed = self.Embedding(sent)
        else:
            embed = self.Embedding(sent)

        # Input Feeding(Optional)
        if self.input_feed:
            embed = torch.cat((embed, feed), axis=-1)
        
        # Dropout before LSTM
        embed = self.Dropout(embed)

        # LSTM
        if train and (not self.input_feed):
            #embed = nn.utils.rnn.pack_padded_sequence(embed, (sent>0).sum(axis=1).tolist(), enforce_sorted=False, batch_first=True)
            outs, hs = self.LSTM(embed, lstm_hs)
            #outs, _ = nn.utils.rnn.pad_packed_sequence(outs, batch_first=True, padding_value=0)
        else:
            outs, hs = self.LSTM(embed, lstm_hs)
        
        return outs, hs

    def reset_buff(self):
        self.embed_buff = None

    def init_weight(self):
        nn.init.uniform_(self.Embedding.weight, -.1, .1)
        init_lstm(self.LSTM)


class Decoder_with_Attention(nn.Module):
    def __init__(self, align, score, vocab_size, embedding_dim, hidden_size, depth, dropout, input_feed, reverse, window_size=10, length=52):
        super(Decoder_with_Attention, self).__init__()

        self.Decoder = Decoder_Rare(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, depth=depth, dropout=dropout, input_feed=input_feed)

        self.input_feed=input_feed

        align = align_check(align)
        if align == 'loc_pred':
            self.Attention = Attention_Local_Predictive(score=score, reverse=reverse, hidden=hidden_size, sigma=window_size//2)
        elif align == 'loc_mono':
            self.Attention = Attention_Local_Monotonic(score=score, hidden=hidden_size, reverse=reverse, window_size=window_size, max_len=length)
        else: # global
            self.Attention = Attention_Global(score=score, reverse=reverse, hidden=hidden_size, length=length)

    def forward(self, enc, dec_sent, feed=None, lstm_hs=None, train=True, word_ind=0):
        dec_out, hs = self.Decoder(dec_sent, feed=feed, lstm_hs=lstm_hs, train=train, word_ind=word_ind)
        
        enc_out, en_len = enc

        att_out = self.Attention(enc=enc_out, dec=dec_out, en_len=en_len, ind=-1 if (train and not self.input_feed) else word_ind)
        #print(att_out.size())
        #print(dec_out.size())
        out = torch.cat((dec_out, att_out), dim=-1)
        return out, hs, att_out

    def reset_buff(self,):
        self.Decoder.reset_buff()
        self.Attention.reset_buff()
    
    def init_weight(self,):
        self.Decoder.init_weight()
        self.Attention.init_weight()


############### AGAIN #####################

# End