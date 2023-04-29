import torch.optim as optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm_

"""
by paper
lr 1e-4
beta1 .9
beta2 .999
weight_decay .01
warmup 10000
total_batch 256
mini_batch ??

in implement
init_lr 5e-5

++
L2?? 5?
"""
class LR_schedule_optim:
    def __init__(self, param, config, ):
        self.param = param
        
        # Initial Set for LR Scheduling
        assert config.lr >= config.init_lr
        
        self.lr = config.lr
        self.warmup = config.warmup #
        self.total_step = config.total_step
        self.init_lr = config.init_lr
        
        self.L2 = config.L2

        self.cur_step = 1
        self.n_count = 0

        self.optimizer = optim.AdamW(param, lr=self.learning_rate(),
                                    betas=(config.beta1, config.beta2),
                                    eps=1e-09, weight_decay=config.weight_decay)
        self.term = int(config.total_batch / config.mini_batch)

    def step_grad(self, loss):
        # current state
        self.n_count += 1

        loss /= self.term
        loss.backward()

        if self.n_count >= self.term:
            self.cur_step += 1

            # LR update
            rate = self.learning_rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate

            # clip Gradient
            clip_grad_norm_(self.param, max_norm=self.L2)
            
            # update N reset
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            self.n_count = 0
            return 1
        return 0

    def zero_grad(self, ):
        self.optimizer.zero_grad(set_to_none=True)
        self.n_count = 0

    def learning_rate(self):
        #BERT uses linear in/decreasing.
        #
        lr = self.init_lr + \
            (self.lr-self.init_lr)* min(self.cur_step, self.warmup) / self.warmup - \
            self.lr * (max(self.cur_step, self.warmup) - self.warmup) / (self.total_step - self.warmup)
        
        return lr

#lr_schedule = lambda step, dim, warmup: (dim**-.5) * min((step+1)**-.5, (step+1) * (warmup**-1.5))
