import torch
import torch.nn as nn
import torch.nn.functional as F

from src.func import score_check

####### FILTER

class _Select(nn.Module):
    """ It only used in 'Local-Mono'. """
    def __init__(self, reverse=True, window_size=10):
        super(_Select, self).__init__()
        #It doesn't need since it has no parameters
        self.reverse = reverse
        
        self.w = window_size

        self.slide = 2*window_size + 1
    
    def forward(self, enc, dec = None):
        """ Shape
        enc : B * E * H      to
              B * D * W * H 
            -> W = 2*window + 1
            -> D = E + 1 (<sos> token)
        """ 
        if self.reverse:
            enc = enc.flip(1)

        if dec != None: ## in Train(get Dec in once)
            # Fit in the length of Enc&Dec output.
            diff = enc.size(1) - dec.size(1)
            enc = F.pad(enc, (0,0, self.w, self.w - diff), 'constant', 0)
        else: ## in Test / Train(each Dec comes)
            enc = F.pad(enc, (0,0, self.w, self.w), 'constant', 0)

        # ReShape Enc 
        enc = enc.unfold(1, self.slide, 1).transpose(2,3)
        return enc



class _Pass(nn.Module):
    """ It is for 'Global'. """
    def __init__(self, reverse=True):
        super(_Pass, self).__init__()
        self.reverse = reverse

    def forward(self, enc):
        if self.reverse:
            enc = enc.flip(1)
        return enc

####### SCORE ########

""" In einsum func,
b <- 'B'atch size
l <- 'L'ength of sentence(only in 'local')
e <- 'E'ncoder length(only in 'global')
d <- 'D'ecoder length(only in 'global')
w <- 'W'indow size
h <- 'H'idden
i <- 'I'ncreased hidden(only in 'con' method)
"""
class _Score_Dot_Local(nn.Module):
    def __init__(self):
        super(_Score_Dot_Local, self).__init__()

    def forward(self, enc, dec, each=False):
        if not each: ## dec passes at once
            score = torch.einsum("blwh, blh -> blw", enc, dec)
        else: ## dec passes one by one
            #enc = enc[:, ind, :, :]
            score = torch.einsum("bwh, bh -> bw", enc, dec)

        return score
    
    def extra_repr(self) -> str:
        return 'mode=\'dot\''
    
    def init_weight(self):
        return # no param exists

class _Score_Dot_Global(nn.Module):
    def __init__(self):
        super(_Score_Dot_Global, self).__init__()

    def forward(self, enc, dec, each=False):
        if not each:
            score = torch.einsum("beh, bdh -> bde", enc, dec)
        else:
            score = torch.einsum("beh, bh -> be", enc, dec)

        return score

    def extra_repr(self) -> str:
        return 'mode=\'dot\''
    
    def init_weight(self):
        return # no param exists


class _Score_Gen_Local(nn.Module):
    def __init__(self, hidden):
        super(_Score_Gen_Local, self).__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
            
    def forward(self, enc, dec, each=False):
        if not each:
            #score = torch.matmul(enc, self.W)
            score = self.W(enc) 
            score = torch.einsum("blwh, blh -> blw", score, dec)
        else:
            #enc = enc[:, ind, :, :].unsqueeze(1)
            score = self.W(enc)
            score = torch.einsum("bwh, bh -> bw", score, dec)

        return score
    
    def extra_repr(self) -> str:
        return f'mode=\'general\', hidden={self.W.size(0)}'
    
    def init_weight(self):
        nn.init.uniform_(self.W.weight, -.1, .1)

class _Score_Gen_Global(nn.Module):
    def __init__(self, hidden):
        super(_Score_Gen_Global, self).__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
            
    def forward(self, enc, dec, each=False):
        score = self.W(enc)
        if not each:
            #score = torch.einsum("beh, bdh -> bde", score, dec)
            score = score.transpose(2,1)
            score = torch.matmul(dec, score)
        else:
            score = torch.einsum("beh, bh -> be", score, dec)
        
        return score
    
    def extra_repr(self) -> str:
        return f'mode=\'general\', hidden={self.W.size(0)}'
    
    def init_weight(self):
        nn.init.uniform_(self.W.weight, -.1, .1)

class _Score_Con_Local(nn.Module):
    def __init__(self, window, hidden):
        super(_Score_Con_Local, self).__init__()
        self.W = nn.parameter.Parameter(torch.randn(2*hidden, hidden))
        self.v = nn.parameter.Parameter(torch.randn(hidden))

        self.window = window

    def forward(self, enc, dec, each=False):
        if not each:
            dec = dec.unsqueeze(2).repeat(repeats=[1, 1, self.window, 1]) # B*L*W*H
            dec = torch.cat((enc, dec), axis=3) # B*L*W*2H
            score = torch.matmul(dec, self.W)   # B*L*W*H
            score = torch.matmul(score, self.v) # B*L*W
        else:
            #enc = enc[:, ind, :, :]
            dec = dec.unsqueeze(1).repeat(repeats=[1, self.window, 1]) # B*W*H
            dec = torch.cat((enc, dec), axis=2) # B*W*2H
            score = torch.matmul(dec, self.W)   # B*W*H
            score = torch.matmul(score, self.v) # B*W
        return score
    
    def extra_repr(self) -> str:
        return f'mode=\'concat\', hidden={self.W.size(0)}'
    
    def init_weight(self):
        nn.init.uniform_(self.W, -.1, .1)
        nn.init.uniform_(self.v, -.1, .1)

class _Score_Con_Global(nn.Module):
    def __init__(self, hidden):
        super(_Score_Con_Global, self).__init__()
        self.W = nn.parameter.Parameter(torch.randn(2*hidden, hidden))
        self.v = nn.parameter.Parameter(torch.randn(hidden))
        
    def forward(self, enc, dec, each=False):
        if not each:
            enc = enc.unsqueeze(1).repeat(repeats=[1, dec.size(1), 1, 1])   # B*D*E*H
            dec = dec.unsqueeze(2).repeat(repeats=[1, 1, enc.size(2), 1])   # B*D*E*H
            dec = torch.cat((enc, dec), axis=3) # B*D*E*2H
            score = torch.matmul(dec, self.W)   # B*D*E*H
            score = torch.matmul(score, self.v) # B*D*E
        else:
            dec = dec.unsqueeze(1).repeat(repeats=[1, enc.size(1), 1])  # B*E*H
            dec = torch.cat((enc, dec), axis=2) # B*E*2H
            score = torch.matmul(dec, self.W)   # B*E*H
            score = torch.matmul(score, self.v) # B*E

        return score
    
    def extra_repr(self) -> str:
        return f'mode=\'concat\', hidden={self.W.size(0)}'

    def init_weight(self):
        nn.init.uniform_(self.W, -.1, .1)
        nn.init.uniform_(self.v, -.1, .1)


class _Score_Loc_Global(nn.Module):
    def __init__(self, length, hidden):
        super(_Score_Loc_Global, self).__init__()
        self.W = nn.parameter.Parameter(torch.randn(hidden, length))

    def forward(self, enc, dec, each=False):
        # Input "each" exists to unify the format.
        score = torch.matmul(dec, self.W[:, :enc.size(1)])
        
        return score

    def extra_repr(self) -> str:
        return f'mode=\'location\', hidden={self.W.size(0)}, length={self.W.size(1)}'

    def init_weight(self):
        nn.init.uniform_(self.W, -.1, .1)

####### ALIGN #######

class _Align_Local_Mono(nn.Module):
    def __init__(self, ):
        super(_Align_Local_Mono, self).__init__()

    def forward(self, enc, score, each=False):
        score = score.softmax(dim=-1)
        if not each:
            # enc   : B*L*W*H
            # score : B*L*W
            out = torch.einsum("blwh, blw -> blh", enc, score)
        else:
            # enc   : B*W*H
            # score : B*W
            out = torch.einsum("bwh, bw -> bh", enc, score)
        return out

class _Align_Local_Pred(nn.Module):
    def __init__(self, hidden, sigma=5):
        super(_Align_Local_Pred, self).__init__()
        
        self.W = nn.parameter.Parameter(torch.randn(hidden, hidden))
        self.v = nn.parameter.Parameter(torch.randn(hidden))
        
        self.window = sigma

    def forward(self, enc, score, en_len, dec, each=False):
        pos = self._position(dec=dec, en_len=en_len, each=each)
        out = self._align(enc=enc, score=score, pos=pos, each=each)
        return out

    def _position(self, dec, en_len, each):
        # dec = B*D*H
        pos = torch.tanh(torch.matmul(dec, self.W))
        pos = torch.sigmoid(torch.matmul(pos, self.v))
        if not each:
            # pos = B*D
            pos = pos.transpose(0,-1) * en_len
            pos = pos.transpose(0,-1)
        else:
            # pos = B
            pos = pos * en_len
        return pos
    
    def _align(self, enc, score, pos, each):
        # enc       = B*E*H
        #print(enc.size())
        score = score * self.__pos_score(pos=pos, enc_len=enc.size(1), each=each)
        if not each:
            out = torch.einsum("beh,bde->bdh", enc, score)
        else:
            out = torch.einsum("beh,be->bh", enc, score)
        return out
    
    def __pos_score(self, pos, enc_len, each):
        if not each:
            # pos = B*D
            pos = pos.unsqueeze(-1).repeat([1,1,enc_len]) # B*D*E
        else:
            # pos = B
            pos = pos.unsqueeze(-1).repeat([1,enc_len]) # B*E
        ind = torch.arange(enc_len).to(pos.device) # E
        out = torch.exp( -.5 * torch.square((ind - pos)/self.window) )
            # not  : B*D*E
            # each : B*E
        return out
    
    def init_weight(self):
        nn.init.uniform_(self.W, -.1, .1)
        nn.init.uniform_(self.v, -.1, .1)

class _Align_Global(nn.Module):
    def __init__(self, ):
        super(_Align_Global, self).__init__()

    def forward(self, enc, score, each=False):
        score = score.softmax(dim=-1)
        if not each:
            # enc   : B*E*H
            # score : B*D*E
            #out = torch.einsum("beh, bde -> bdh", enc, score)
            out = torch.matmul(score, enc)
            # out   : B*D*H
        else:
            # enc   : B*E*H
            # score : B*E
            out = torch.einsum("beh, be -> bh", enc, score)
            # out   : B*H
        return out

####### ATTENTION #######

class Attention_Local_Monotonic(nn.Module):
    def __init__(self, score, hidden=1000, reverse=True, window_size=10, max_len=52):
        super(Attention_Local_Monotonic, self).__init__()
        
        self.Filter = _Select(reverse=reverse, window_size=window_size)
        
        self.enc_buff = None

        mode = score_check(score)
        #print(f"mode : {mode}")
        if mode == 'dot':
            self.Score = _Score_Dot_Local()
        elif mode == 'gen':
            self.Score = _Score_Gen_Local(hidden=hidden)
        elif mode == 'con':
            self.Score = _Score_Con_Local(window=2*window_size+1, hidden=hidden)
        else: # mode == 'loc'
            ValueError("mode 'score' is not adequate.")

        self.Align = _Align_Local_Mono()

    def forward(self, enc, dec, en_len=None, ind=-1):
        """ Input
        enc : B * E * H      will be re-shaped into
            : B * E * W * H  (W = 2*D + 1).
            After that it is "attended" by
        dec : B * D  * H
            if each = False
            : B * H
            elif each = True
        ind : -1 when dec is (B*D*H)
            : k when dec is k'th (B*H) 
        """
        if ind == -1:   ## all Decs pass at once
            enc = self.Filter(enc=enc, dec=dec)
            #print(enc.size())
            score = self.Score(enc=enc, dec=dec, each=False)
            #print(score.size())
            out = self.Align(enc=enc, score=score, each=False)

        else:           ## each Dec passes separately
            if self.enc_buff == None:
                self.enc_buff = self.Filter(enc=enc, dec=None)
            
            # Enc/Dec length may not be fit
            ind = min(ind, self.enc_buff.size(1))

            enc = self.enc_buff[:, ind, :, :].unsqueeze(1)
            #print(enc.size())
            score = self.Score(enc=enc, dec=dec, each=False)
            #print(score.size())
            out = self.Align(enc=enc, score=score, each=False)
        
        return out
    
    def reset_buff(self):
        self.enc_buff = None
    
    def init_weight(self):
        self.Score.init_weight()

class Attention_Local_Predictive(nn.Module):
    def __init__(self, score='dot', reverse=True, hidden=1000, sigma=5):
        super(Attention_Local_Predictive, self).__init__()
    
        self.Filter = _Pass(reverse=reverse)
        
        self.enc_buff = None

        mode = score_check(score)
        if mode == 'dot':
            self.Score = _Score_Dot_Global()
        elif mode == 'gen':
            self.Score = _Score_Gen_Global(hidden=hidden)
        elif mode == 'con':
            self.Score = _Score_Con_Global(hidden=hidden)
        else: # mode == 'loc'
            ValueError("mode 'score' is not adequate.")

        self.Align = _Align_Local_Pred(hidden=hidden, sigma=5)

    def forward(self, enc, dec, en_len, ind=-1):
        if ind == -1:
            enc = self.Filter(enc=enc)
            score = self.Score(enc=enc, dec=dec, each=False)
            out = self.Align(enc=enc, score=score, dec=dec, en_len=en_len, each=False)

        else:
            if self.enc_buff == None:
                self.enc_buff = self.Filter(enc=enc)
            score = self.Score(enc=self.enc_buff, dec=dec, each=False)
            out = self.Align(enc=enc, score=score, en_len=en_len, dec=dec, each=False)
        return out

    def reset_buff(self):
        self.enc_buff = None

    def init_weight(self):
        self.Score.init_weight()
        self.Align.init_weight()

class Attention_Global(nn.Module):
    def __init__(self, score='dot', reverse=True, hidden=1000, length=52):
        super(Attention_Global, self).__init__()
        
        self.Filter = _Pass(reverse=reverse)
        
        self.enc_buff = None

        mode = score_check(score)
        if mode == 'dot':
            self.Score = _Score_Dot_Global()
        elif mode == 'gen':
            self.Score = _Score_Gen_Global(hidden=hidden)
        elif mode == 'con':
            self.Score = _Score_Con_Global(hidden=hidden)
        else: # mode == 'loc'
            self.Score = _Score_Loc_Global(length=length, hidden=hidden)

        self.Align = _Align_Global()

    def forward(self, enc, dec, en_len=None, ind = -1):
        """ Shape
        enc : B * E * H
            It is "attended" by
        dec : B * D * H
            if each = False
            : B * H
            elif each = True
        """
        if ind == -1:   ## all Decs pass at once
            enc = self.Filter(enc)
            score = self.Score(enc, dec)
            out = self.Align(enc, score)

        else:           ## each Dec passes separately
            if self.enc_buff == None:
                self.enc_buff = self.Filter(enc=enc)
            score = self.Score(self.enc_buff, dec, each=False)
            out = self.Align(enc, score, each=False)
        
        return out

    def reset_buff(self):
        self.enc_buff = None

    def init_weight(self):
        self.Score.init_weight()