import torch
import torch.nn as nn
import random
import numpy as np

seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

x=torch.ones(5)
drop=nn.Linear(5,1)
print(x)
drop.train()
print(drop(x))   # one random mask
print(drop(x))
print(drop(x))