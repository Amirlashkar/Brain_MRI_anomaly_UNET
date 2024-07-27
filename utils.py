from typing import Tuple
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle as pkl
import functions
from torch.utils.data import Dataset
from torch._prims_common import DeviceLikeType
import torch.nn as nn
import torch
import os


class ImageDataset(Dataset):
    """
    Making data accessible
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
