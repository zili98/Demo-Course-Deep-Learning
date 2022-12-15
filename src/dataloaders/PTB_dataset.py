import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, sampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from yacs.config import CfgNode
from dataloaders.collate_fn import BatchBalanceCollate

class PTBDataset(Dataset):
    def __init__(self, data, downsample=None, normalize=True, transform=None, two_class=False):
        self._data = data.item()['ECG_signal']
        self._labels = data.item()['label']
        self._transform = transform
        if downsample:
          self._data = self._data[:, ::downsample, :]
        
        if normalize:
          std_ = self._data.std(axis=1, keepdims=True)
          std_[std_ == 0] = 1.0
          self._data = (self._data - self._data.mean(axis=1, keepdims=True)) / std_
        
        self._data = np.transpose(self._data, (0,2,1))

        if two_class:
            for i in range(len(self._labels)):
                if self._labels[i] in [0,1,2,3]:
                    self._labels[i] = 0
                else:
                    self._labels[i] = 1

    def __len__(self):
        return len(self._labels)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg = self._data[idx,:]
        # ecg = ecg.reshape(1, -1)
        label = self._labels[idx]

        sample = {'label':label, 'ecg':ecg}

        if self._transform:
            sample = self._transform(sample)

        return sample

class PTBDataLoader():
    def __init__(self, cfg,
                train_transform=None, val_transform=None, test_transform=None,
                downsample=None, normalize=True, two_class=False):

        self._cfg = cfg
        self._train_transform = train_transform
        self._val_transform = val_transform
        self._test_transform = test_transform
        self._downsample = downsample
        self._normalize = normalize
        self._two_class=two_class
        self._collate_fn = BatchBalanceCollate()
    
    def _set_up_dataset(self):
        train_data = np.load(self._cfg.DATASET.TRAIN_FILE, allow_pickle=True)
        val_data = np.load(self._cfg.DATASET.VAL_FILE, allow_pickle=True)
        test_data = np.load(self._cfg.DATASET.TEST_FILE, allow_pickle=True)

        self._ds_train = PTBDataset(train_data, downsample=self._downsample, normalize=self._normalize, transform=self._train_transform, two_class=self._two_class)
        self._ds_val = PTBDataset(val_data, downsample=self._downsample, normalize=self._normalize, transform=self._val_transform, two_class=self._two_class)
        self._ds_test = PTBDataset(test_data, downsample=self._downsample, normalize=self._normalize, transform=self._test_transform, two_class=self._two_class)

    def create_dataloaders(self):
        self._set_up_dataset()
        train_loader = DataLoader(self._ds_train, 
                                  batch_size=self._cfg.DATALOADER.BATCH_SIZE,
                                  shuffle=True, generator=torch.Generator(device='cuda'),
                                  collate_fn=self._collate_fn,
                                )
        val_loader = DataLoader(self._ds_val, 
                                  batch_size=self._cfg.DATALOADER.BATCH_SIZE,
                                  shuffle=True, generator=torch.Generator(device='cuda'),
                                  collate_fn=self._collate_fn,
                                )
        test_loader = DataLoader(self._ds_test, 
                                  batch_size=1,
                                  shuffle=False, generator=torch.Generator(device='cuda'),
                                  collate_fn=self._collate_fn,
                                )

        dataloaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}
        return dataloaders

