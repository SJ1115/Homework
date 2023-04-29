class Config:
    """
    Config Setting :
        Model Setting at BERT_base
        + Data Setting with my own.
    """
    ## Data Config

    tokenizer = None
    vocab_size = 30000
    n_segment = 2
    max_len = 128
    padding_idx = 0
    masking_idx = 1

    ## Model Config ##

    hidden = 512
    FF_hidden = 2048
    n_head = 6
    n_layer = 6

    norm_eps = 1e-5

    dropout = .1
    is_pre_norm = False

    ## Train Config ##

    init_lr = 5e-5
    lr = 1e-4
    beta1 = .9
    beta2 = .999
    weight_decay = .01
    L2 = 5
    warmup = 10000
    total_batch = 256
    total_step = 1000000
    
    ## Run Config ##
    
    device = 'cpu'
    mini_batch = 1
    use_tqdm = True
    use_board = False
    num_workers = 0 #???
    
    src_dir = None
    filename = None
    
    ## Task(GLUE) Config ##
    
    task_lr = 3e-4
    epoch = 4
    task_batch = 128
    
    def __init__(self,):
        self.task_cls = None
        self.task_dir = None
        self.task_is_regression = False
    
    def set_task(self,task):
        task = task.lower()
        assert task in ("cola", "sst2", "mrpc", "stsb",
                 "qqp", "mnli", "mnlim", "qnli", "rte", "wnli")
        if task in ("qqp", "qnli", "sst2", "cola", "mrpc",
                    "rte", "wnli"):
            self.task_cls = 2
        elif task in ('mnli','mnlim'): # ????????
            self.task_cls = 3
        elif task in ('stsb',):
            self.task_cls = 1
            self.task_is_regression = True

        self.task_dir = [f"data/task/processed/{task}_{mode}.json" for mode in ('train', 'valid')]
        
        if task == 'mnlim':
            self.task_dir[0] = self.task_dir[0][:24] + self.task_dir[0][25:]
            

class Config_mini(Config):
    """
    I Only changed the model size setting
    """
    hidden = 128
    FF_hidden = 512
    n_head = 2
    n_layer = 2
    
    total_batch = 128


class Config_toy(Config_mini):
    """
    To run small, I Additionally reduced warmup, batch, and steps
    """
    ## Train Config ##
    warmup = 500
    total_batch = 32
    total_step = 10000
    
