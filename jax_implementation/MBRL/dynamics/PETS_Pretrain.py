import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import Dataset
def load_data(data_path):
    data = np.load(data_path)

    obs = data["obs"]                    # (B, H, 96)
    act = data["applied_action"]         # (B, H, 4)
    next_obs = data["next_obs"]          # (B, H, 96)
    rew = data["reward"][..., None]      # (B, H, 1)

    x = np.concatenate([obs, act], axis=-1)          # (B, H, 100)
    y = np.concatenate([next_obs - obs, rew], axis=-1)  # (B, H, 97)
    return jp.array(x), jp.array(y), jp.array(rew)

def collate_fn(batch):
    x,y = zip(*batch)
    return jp.array(x, dtype=jp.float32), jp.array(y, dtype=jp.float32)
class data_load(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        xval = self.x[idx]
        yval = self.y[idx]
        
        return jp.array(xval, dtype=jp.float32), jp.array(yval, dtype=jp.float32)
    


if __name__ == "__main__":
    path = "jax_implementation/MBRL/dyn_data/pets_pretrain_envB1024_T10000_pid_noisy_seed0.npz"
    x, y, rews = load_data(path)
    B, H = x.shape[0], x.shape[1]
    rng = np.random.default_rng(0)

    perm = rng.permutation(B)
    n_train = int(0.9 * B)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]
    
    train_dataset = data_load(train_x, train_y)
    test_dataset = data_load(test_x, test_y)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)
    

