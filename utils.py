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


class PXLoss(nn.Module):
    def __init__(self, device:DeviceLikeType) -> None:
        super(PXLoss, self).__init__()

        self.device = device

    def forward(self, predictions, targets):
        targets = targets.cpu().detach().numpy()
        mask = torch.tensor(functions.segment_brain(targets)).to(self.device)
        targets = torch.tensor(targets).to(self.device)
        diff = torch.abs(predictions - targets)
        loss = torch.mean(torch.pow(diff, 2) * mask)

        return loss
