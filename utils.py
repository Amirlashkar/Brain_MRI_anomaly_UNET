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


class Checkpointer:
    def __init__(
        self,
        model:torch.nn.Module,
        scaler:StandardScaler,
        thresholds:Tuple[float, float]
        ) -> None:

        self.model = model
        self.scaler = scaler
        self.thresholds = thresholds

        self.checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
        self.check_folder()

        self.last_state_path = None

    def check_folder(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def save(self, epoch:int, epoch_loss:float, desc:str):
        """
        Main function to save some objects and variables

        epoch: number of current epoch
        epoch_loss: average loss of current epoch
        desc: description of model architecture, and etc.
        """

        now = str(datetime.now()).replace(" ", "_")
        name = f"{now}_epoch:{epoch}_loss:{'{:.3f}'.format(epoch_loss)}"
        name = name.replace(":", "_").replace("-", "_")
        path = os.path.join(self.checkpoint_path, name)
        os.makedirs(path)

        self.last_state_path = os.path.join(path, "state.pth")
        state_dict = {
            "model_arch": str(self.model),
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "thresholds": self.thresholds,
            "description": desc,
        }

        torch.save(state_dict, self.last_state_path)
