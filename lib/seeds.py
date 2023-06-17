'''
Date: 2021-01-13 16:34:01
LastEditTime: 2021-01-13 16:34:49
Description: Set random seeds
FilePath: /DMGAN/lib/seeds.py
'''
import os
import torch
import numpy as np
import random

def seed(seed=43):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False