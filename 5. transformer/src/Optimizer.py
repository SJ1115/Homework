import torch.optim as optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm_


class LRScheduler:
    def __init__(self, model, lr=1, mini_batch=32, total_batch=4096, warmup=4000, L2=1):
        self.model = model
        
        # Initial Set for LR Scheduling
        self.dim = model.model_dim ** -.5 #/ (512 / batch_size)
        self.lr = lr
        self.cur_step = 1
        self.warmup = warmup #* (512 / batch_size)
        self.L2 = L2

        self.n_step = 0

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate(),
                                    betas=(0.9, 0.997), eps=1e-09)
        self.term = int(total_batch / mini_batch)

    def step_grad(self, loss, n_vocab):
        # current state
        self.n_step += 1

        loss /= self.term
        loss.backward()

        if self.n_step >= self.term:
            self.cur_step += 1

            # LR update
            rate = self.learning_rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate

            # clip Gradient
            clip_grad_norm_(self.model.parameters(), max_norm=self.L2)
            
            # update
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # reset
            self.n_step = 0
            
            return 1
        return 0

    def zero_grad(self, ):
        self.optimizer.zero_grad(set_to_none=True)
        self.n_step = 0

    def learning_rate(self):
        #lr = self.lr * self.dim
        #lr *= min(1.0, self.cur_step / self.warmup)
        #lr *= max(self.cur_step, self.warmup) ** -0.5
        #
        lr = self.lr * self.dim * min( self.cur_step**-.5 , self.cur_step * (self.warmup**-1.5))
        return lr

#lr_schedule = lambda step, dim, warmup: (dim**-.5) * min((step+1)**-.5, (step+1) * (warmup**-1.5))
