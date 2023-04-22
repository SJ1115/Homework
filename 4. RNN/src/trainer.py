import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score as BLEU

class Trainer:
    def __init__(self, model, criterion, optimizer, id_to_word, data, batch_size, device, l2=5):
        self.device = device
        self.model = model.to(device)
        self.model.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.id_to_word = id_to_word
        self.L2 = l2

        train_in, train_out, test_in, test_out = data
        #train_in = train_in.to(device);    train_out = train_out.to(device)
        #test_in  = test_in.to(device);     test_out = test_out.to(device)

        self.trainLoader = DataLoader(list(zip(train_in, train_out)), batch_size=batch_size, shuffle=True, num_workers=4)
        self.testLoader  = DataLoader(list(zip(test_in,  test_out )), batch_size=1, shuffle=False, num_workers=4)

        self.loss_list = []
        self.verbose_flag = False

    def train(self, epoch, decreasing_point, filename=None, early_stopping = 0, show_batches = 1):
        """
        args:
            epoch : int, epoch size
            decreasing_point : int, epoch number when to start halving LR
            filename : str, saving point for model. None(default) means not saving model.
            early_stopping : int, patience_term in Early Stopping. 0(default) means there is no early stopping.
            show_batches : int, it is for tqdm()'s printing term. large int means long-term, so there would be faster in training.
        """
        if self.verbose_flag:
            start = time()
            timeiter = tqdm(total = len(self.trainLoader)*epoch, position=0, leave=True)

        # Lr Scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=.5)
        # Set for checking progress
        current_loss = 0
        cut = 0

        # For early stopping
        # stop_cnt = -1 : no stopping,
        #          = k  : stop after k times non-improved epoches 
        stop_cnt = early_stopping
        stop_cnt -= not bool(stop_cnt)
        current_cnt = stop_cnt
        min_ppl = 1e+7

        # Set Train Mode
        self.model.train()
        torch.enable_grad()
        
        ### Train ###
        e = 0

        # Load from checkpoint
        if not filename == None:
            try:
                from_file = torch.load(filename, map_location=self.device)
                temp_e = from_file["epoch"]
                
                if temp_e > e:
                    self.model = from_file["model"]
                    if self.verbose_flag:
                        timeiter.update((temp_e-e)*len(self.trainLoader))
                    e = temp_e
                del from_file

                if e >= decreasing_point:
                    for i in range(e - decreasing_point + 1):
                        self.optimizer.zero_grad()
                        self.optimizer.step()
                        scheduler.step()
            except:
                0
        
        while e < epoch:
            # LR Decreasing
            if e >= decreasing_point:
                scheduler.step()

            for data in self.trainLoader:
                seq_in, seq_out = data
                seq_in = seq_in.to(self.device); seq_out = seq_out.to(self.device)
                
                # reset gradient
                self.optimizer.zero_grad(set_to_none=True)

                # for/backward
                outputs = self.model.forward(seq_in, seq_out[:,:-1])
                loss = self.criterion(outputs, seq_out[:, 1:outputs.size(2)+1])
                loss.backward()

                # 
                #del data

                # update
                nn.utils.clip_grad_norm_(self.model.parameters(), self.L2)
                self.optimizer.step()

                # check progress
                current_loss += loss.item()
                cut += 1
                (lambda x: timeiter.update(1) if x else 0)(self.verbose)
                
                if cut % show_batches == 0:
                    showing_loss = current_loss / show_batches
                    self.loss_list.append(showing_loss)
                    current_loss = 0.0
                    cut = 0

                    

                    if self.verbose_flag:
                    # print statistics        
                        timeiter.set_description(f"loss : {showing_loss: .3f}")
            
            e += 1

            # Save model checkpoint
            if not filename == None:
                torch.save({"model": self.model, "epoch": e}, filename)

            # Test for early stopping : Based on PPL
            if early_stopping:
                ppl, _ = self.test(verbose=False)
                if ppl < min_ppl:
                    min_ppl = ppl
                    current_cnt = stop_cnt
                else:
                    current_cnt -= 1
                if not current_cnt:
                    break
            
        return

    def test(self, verbose=None):
        self.model.eval()

        # for BLEU test
        candidates = []
        references = []

        # for PPL test
        tot_loss = 0.0

        verbose = self.verbose_flag if verbose==None else verbose
        timeiter = tqdm(self.testLoader, desc="Testing...") if verbose else self.testLoader
        
        with torch.no_grad():
            for data in timeiter:
                seq_in, seq_out = data
                seq_in = seq_in.to(self.device); seq_out = seq_out.to(self.device)
                
                # forward : for Perplexity
                outputs = self.model.forward(seq_in, seq_out[:,:-1])
                loss = self.criterion(outputs, seq_out[:, 1:outputs.size(2)+1])
                tot_loss += loss.item()


                # predict : for BLEU
                words, _ = self.model.predict(seq_in, max_len=seq_in.shape[1]-1, device=self.device)

                answers = seq_out[:, 1:]

                candidates += [[self.id_to_word[int(word)] for word in sent if word > 0] for sent in words]
                references += [[[self.id_to_word[int(word)] for word in sent if word > 0]] for sent in answers]
                
        # Reset State
        self.model.train()

        bleu_score = BLEU(candidate_corpus=candidates, references_corpus=references)
        ppl_score  = tot_loss/len(timeiter)
        
        return ppl_score, bleu_score


    def plot(self):
        if len(self.loss_list)==0:
            raise ValueError("Train Model First")
        
        fig = plt.plot(self.loss_list)

        plt.show(fig)

        return

    def verbose(self, make = None):
        if make == None:
            self.verbose_flag = not self.verbose_flag
        elif make in (True, False):
            self.verbose_flag = make
        else:
            ValueError("Input must be in (True, False, or None)")

##### END #####