import math
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

def Loader(full_dataset, train_size, valid_size, test_size, batch_size,r_seed):
    torch.manual_seed(r_seed)
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dl, valid_dl, test_dl
