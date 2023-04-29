import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score as bleu
import dill

from tqdm import tqdm
from src.Func import src_to_lines
from src.Optimizer import LRScheduler

class Trainer:
    def __init__(self, tokenizer, model, criterion, lr=1, mini_batch=32, total_batch=4096, device='cpu', warmup=4000, l2=1):
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        self.model.device = device
        self.criterion = criterion
        self.optimizer = LRScheduler(self.model, lr=lr, mini_batch=mini_batch, total_batch=total_batch, warmup=warmup, L2 = l2)
        self.batch_size = mini_batch

        self.loss_list = []
        self.verbose = False
        self.timeiter = None

        self.device = device

        self.Loader = None
                
    def train(self, train_cnt, src_lst, lr=2, max_len=128, filename=None, show_batches = 1, num_workers=0):
        """
        args:
            train_cnt : int, total train steps. epoch will be repeated until satisfying the total steps.
            warmup : int, parameter in lr scheduling. default 4000
            filename : str, saving point for model. None(default) means not saving model.
            show_batches : int, it is for tqdm()'s printing term. large int means long-term, so there would be faster in training.
        """

        if self.verbose:
            self.timeiter = tqdm(total = train_cnt, position=0, leave=True)
        
        # Load from checkpoint
        current_cnt = self._load_model(filename)

        self.timeiter.update(current_cnt) if self.verbose else 0
        self.optimizer.cur_step = current_cnt
                
        
        flag = True

        loss_cut = 1e+9
        
        # reset gradient
        self.optimizer.zero_grad()

        # Set Train Mode
        self.model.train()
        torch.enable_grad()
        
        ### Train ###
        while flag:
            np.random.shuffle(src_lst)

            for src_pair in src_lst:
                self._load_data(src_pair, num_workers=num_workers)

                current_cnt = self._train_from_loader(cnt=current_cnt, total_cnt=train_cnt, max_len=max_len, filename=filename, show_batches=show_batches)

                temp_loss = np.mean(self.loss_list)
                    
                if temp_loss <= loss_cut:
                    self._save_model(filename, current_cnt)
                    loss_cut = temp_loss
                else:
                    self._load_model(filename)
                    self.optimizer.cur_step = current_cnt
                #self.loss_list = []

                if current_cnt >= train_cnt:
                    flag = False
                    break

        return

    def test(self, src, verbose=None, beam=False, max_len=128, margin_len=50, num_workers=0):
        """
        args:
            train_cnt : int, total train steps. epoch will be repeated until satisfying the total steps.
            warmup : int, parameter in lr scheduling. default 4000
            filename : str, saving point for model. None(default) means not saving model.
            show_batches : int, it is for tqdm()'s printing term. large int means long-term, so there would be faster in training.
        """
        verbose = self.verbose if verbose == None else verbose
        
        self._load_data(src, batch_size = 1, num_workers=num_workers)

        self.timeiter = tqdm(self.Loader, position=0, leave=True) if verbose else self.Loader

        predict_len = max_len + margin_len
        
        # for BLEU test
        candidates = []
        references = []

        self.model.eval()
        with torch.no_grad():
            for data in self.timeiter:
                problem, answer = data

                problem = torch.tensor([line.ids for line in self.tokenizer.encode_batch(problem)]).to(self.device)
                answer  = torch.tensor([line.ids for line in self.tokenizer.encode_batch(answer )]).to(self.device)
                
                problem = problem[:, :max_len]
                answer  = answer[:, 1:predict_len]

                if beam:
                    my_ans = self.model.predict(problem, max_len = predict_len, device=self.device)
                else:
                    my_ans = self.model.New_predict(problem, max_len = predict_len, device=self.device)

                candidates += [[self.tokenizer.id_to_token(i) for i in sent if i > 0] for sent in my_ans]
                references += [[[self.tokenizer.id_to_token(i) for i in sent if i > 0]] for sent in answer]
        
        self.model.train()
        score = bleu(candidate_corpus=candidates, references_corpus=references)

        return score

    def _train_from_loader(self, cnt, total_cnt, max_len, term=1, filename=None, show_batches=1):
        
        # Set for checking progress
        current_loss = 0
        cut = 0
        
        for data in self.Loader:
            seq_in, seq_out = data
            seq_in = torch.tensor([line.ids for line in self.tokenizer.encode_batch(seq_in)]).to(self.device)
            seq_out = torch.tensor([line.ids for line in self.tokenizer.encode_batch(seq_out)]).to(self.device)
            
            # Cut too long sentences, to prevent GPU MEM overflow.
            seq_in  = seq_in[:, :max_len]
            seq_out = seq_out[:, :max_len]

            
            # for/backward
            outputs = self.model.forward(seq_in, seq_out[:, :-1]).softmax(dim=1)
            loss = self.criterion(outputs, seq_out[:, 1:]) / term # Size of outputs might have been cut.

            if self.verbose:
            # print statistics        
                self.timeiter.set_description(f"lr :{self.optimizer.learning_rate()*1000:.2f}e-3, loss :{loss.item(): .3f}")  

            # update
            #nn.utils.clip_grad_norm_(self.model.parameters(), self.L2)
            n_vocab = int((seq_in > 0).sum() + (seq_out > 0).sum())
            
            if self.optimizer.step_grad(loss, n_vocab):

                cnt += 1
                self.loss_list.append(loss.item())
                if len(self.loss_list) >= 100:
                    self.loss_list.pop(0)

                self.timeiter.update(1) if self.verbose else 0
                self.loss_list.append(loss.item())

            if cnt >= total_cnt:
                break

        return cnt

    def _load_data(self, src_pair, batch_size=0, num_workers=4):
        if not batch_size:
            batch_size = self.batch_size
        src_in, src_out = src_pair

        data_in = src_to_lines(src_in)
        data_out = src_to_lines(src_out)
        
        self.Loader = DataLoader(list(zip(data_in, data_out)), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    def _load_model(self, filename):
        if filename == None:
            return 1
        
        try:
            from_file = torch.load(filename, pickle_module=dill)
            #self.model.state_dict(from_file["model"]).to(self.device)
            self.model = from_file["model"].to(self.device)
            self.tokenizer = from_file["tokenizer"]
            self.optimizer = from_file["optimizer"]
            cnt = from_file["step"]

            del from_file
        except:
            cnt = 1
            
        return cnt
    
    def _save_model(self, filename, cnt):
        if filename == None:
            return None

        torch.save({"model": self.model, 
                        "step": cnt,
                        "optimizer": self.optimizer,
                        "tokenizer" : self.tokenizer},
                filename, pickle_module=dill)
        return None
    
    def set_verbose(self, verbose=True):
        self.verbose = verbose

    def device_to(self, device):
        self.device = device
        self.model.to(device)


###### LR Scheduler ######

lr_schedule = lambda step, dim, warmup: (dim**-.5) * min((step+1)**-.5, (step+1) * (warmup**-1.5))
