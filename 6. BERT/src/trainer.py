import os
import numpy as np
import json

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.optimizer import LR_schedule_optim
from src.preprocess import read_instance, custom_collate
from src.model import BERT_GLUE

class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        
        self.MLM_criterion = nn.CrossEntropyLoss()
        self.NSP_criterion = nn.CrossEntropyLoss()
        
        self.optimizer = LR_schedule_optim(model.parameters(), config)
        
        self.mini_batch = config.mini_batch
        
        self.Loader = None
        
        self.num_workers = config.num_workers
        self.device = config.device
        
        self.use_tqdm = config.use_tqdm
        self.use_board = config.use_board
        
        self.loss_lst = []
               
    def pre_train(self, config):
        # use all file in directory
        src_lst = os.listdir(config.src_dir)
        src_lst = [config.src_dir + src for src in src_lst]
        
        # Load Model
        step = self._load_model(config.filename)
        
        # Set Checker
        if self.use_tqdm:
            self.timecheck = tqdm(total=config.total_step, position=0, leave=True)
            self.timecheck.update(step)
        #if self.use_board:
            #self.summary = SummaryWriter()
        
        # Set Train Mode
        self.model.train()
        self.optimizer.zero_grad()
        torch.enable_grad()
        
        # Train
        while step < config.total_step:
            np.random.shuffle(src_lst)
            for src in src_lst:
                self._load_data_for_pre_train(src)
                
                step = self._pre_train_within_loader(step, config.total_step)
                
                self._save_model(config.filename, step)
                if step >= config.total_step:
                    break
        if self.use_board:
            self.summary.close()
    
    def _load_data_for_pre_train(self, src,):
        with open(src, 'r') as f:
            data = json.load(f)
        self.Loader = DataLoader(data, collate_fn=custom_collate, shuffle=True, num_workers=self.num_workers)
        return
    
    def _pre_train_within_loader(self, step, total_step):
        for data in self.Loader:
            tokens, labels, positions, segments, is_random = data
            tokens, labels, positions, segments, is_random = tokens.to(self.device), labels.to(self.device), positions.to(self.device), segments.to(self.device), is_random.to(self.device)
            # forward
            cls, wrd = self.model.forward(tokens, segments, positions)
            
            loss = self.MLM_criterion(wrd, labels) + self.NSP_criterion(cls, is_random)
            
            self.loss_lst.append(loss.item())
            if len(self.loss_lst)>100:
                self.loss_lst.pop(0)
                    
            if self.optimizer.step_grad(loss):
                step += 1
                if self.use_tqdm:
                    self.timecheck.update()
                    self.timecheck.set_description(f"loss :{np.mean(self.loss_lst):.3f}")
                if self.use_board:
                    self.summary.add_scalar('loss', np.mean(self.loss_lst), step)
                    self.summary.add_scalar('LR', self.optimizer.learning_rate(), step)
                    
            if step >= total_step:
                break
        
        return step
    
    def _load_model(self, filename):
        try:
            with open(filename, "rb") as f:
                load = torch.load(f, map_location=self.device)
            self.model.load_state_dict(load['model'])
            self.optimizer.optimizer.load_state_dict(load['optim'])
            step = load['step']
            self.optimizer.cur_step = step
        except:
            step = 0
        return step
    
    def _save_model(self, filename, step):
        if filename:
            with open(filename, "wb") as f:
                torch.save({
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.optimizer.state_dict(),
                    "step" : step
                }, f)
    
    
class Tuner:
    def __init__(self, model, config):
        self.Model = BERT_GLUE(model, config).to(config.device)
        self.Optim = Adam(self.Model.parameters(), config.task_lr)
        
        self.is_reg = config.task_is_regression
        if self.is_reg:
            self.LossF = nn.MSELoss()
        else:
            self.LossF = nn.CrossEntropyLoss()
        
        self.config = config
        self.device = config.device
        
        self.trainLoader = None
        self.validLoader = None
        self._load()
        
        self.use_tqdm = config.use_tqdm
        self.loss_lst = []
    
    def tune(self, ):
        if self.use_tqdm:
            check = tqdm(total=self.config.epoch * len(self.trainLoader))
        
        self.Model.train()
        torch.enable_grad()
        
        for e in range(self.config.epoch):
            for data in self.trainLoader:
                self.Optim.zero_grad()
                
                line, seg, label = data
                line, seg, label = line.to(self.device), seg.to(self.device), label.to(self.device)

                if self.is_reg:
                    label = label.reshape(-1, 1)

                cls  = self.Model(line, seg)
                loss = self.LossF(cls, label)
                
                loss.backward()
                l = loss.item()
                
                self.Optim.step()
                
                self.loss_lst.append(l)
                if self.use_tqdm:
                    check.update(1)
                    check.set_description(f"loss :{l:.3f}")
                
        return
    
    def test(self, ):
        if self.is_reg:
            answ, pred = [], []
        else:
            correct, total = 0, 0
        
        self.Model.eval()
        with torch.no_grad():
            for data in self.validLoader:
                line, seg, label = data
                line, seg, label = line.to(self.device), seg.to(self.device), label.to(self.device)

                out  = self.Model(line, seg)
                
                if self.is_reg:
                    answ += label.tolist()
                    pred += out.squeeze(1).tolist()
                else:
                    _, pred = torch.max(out.data, 1)
                    total += label.size(0)
                    correct += (pred == label).sum().item()

        if self.is_reg:
            r, _ = spearmanr(pred, answ)
            print(f'Correlation of the Model on the {len(self.validLoader.dataset)} set: {50 * r + 50:.2f} %') 
            return .5 * r + .5
        else:
            print(f'Accuracy of the Model on the {len(self.validLoader.dataset)} set: {100 * correct / total:.2f} %') 
            return correct/total
    
    def plot(self):
        if len(self.loss_list)==0:
            raise ValueError("Train Model First")
        

        if self.dev:
            fig, ax_loss = plt.subplots()
            ax_loss.plot(self.loss_list, label='y1', color='green')
            ax_loss.set_ylabel('Loss')

            ax_right = ax_loss.twinx()
            ax_right.plot([dev*100 for dev in self.dev_list], label='Dev', color='orange')
            ax_right.set_ylabel('score(%)')
        
        else:
            fig = plt.plot(self.loss_list)

        plt.show(fig)

    def _load(self, ):
        train_dir, valid_dir = self.config.task_dir
        
        with open(train_dir, 'r') as f:
            loaded = json.load(f)
        self.trainLoader = DataLoader(list(zip(torch.tensor(loaded['lines']), torch.tensor(loaded['segments']), torch.tensor(loaded['labels']))), batch_size=self.config.task_batch, shuffle=True)

        with open(valid_dir, 'r') as f:
            loaded = json.load(f)
        self.validLoader = DataLoader(list(zip(torch.tensor(loaded['lines']), torch.tensor(loaded['segments']), torch.tensor(loaded['labels']))), batch_size=self.config.task_batch)

        return