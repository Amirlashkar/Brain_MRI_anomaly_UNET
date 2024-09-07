from typing import Tuple
from constants import *
from datetime import datetime
import functions
from torch.utils.data import Dataset
from torch._prims_common import DeviceLikeType
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch
import os


class ImageDataset(Dataset):
    """
    Making data accessible
    """

    def __init__(self, data:torch.Tensor):
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
        # Mask
        targets = targets.cpu().detach().numpy()
        masks = [torch.tensor(functions.masking(t.squeeze(0))).to(self.device) for t in targets]
        masks = torch.stack(masks).unsqueeze(1)

        # Main part to compare two batches
        metric = StructuralSimilarityIndexMeasure(data_range=1.).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)


        # This loop is used not to let spots with high loss happen on reconstruction
        losses = []
        for ind, target in enumerate(targets):
            t_ = target.squeeze(0)
            p_ = predictions[ind][0].squeeze(0)

            for i in range(0, SHAPE[0], SHAPE[0]):
                for j in range(0, SHAPE[0], SHAPE[0]):
                    t_patch = t_[i:i+PATCH_DIM, j:j+PATCH_DIM]
                    p_patch = p_[i:i+PATCH_DIM, j:j+PATCH_DIM]

                    ssim = metric(p_patch.unsqueeze(0).unsqueeze(0), t_patch.unsqueeze(0).unsqueeze(0))
                    ssim = 1 - ssim
                    diff = torch.abs(p_patch - t_patch)
                    diff = torch.mean(diff)

                    if not torch.isnan(ssim):
                        loss = SSIM_ALPHA * ssim + (1 - SSIM_ALPHA) * diff
                        losses.append(loss)

        patch_loss = torch.mean(torch.stack(losses))

        ssim = metric(predictions, targets)
        ssim = 1 - ssim
        diff = torch.abs(predictions - targets)
        diff = torch.mean(diff)
        # packing up all factors together
        loss = SSIM_ALPHA * ssim + ((1 - SSIM_ALPHA) * .75) * diff +  ((1 - SSIM_ALPHA) * .25) * patch_loss

        return loss

class Checkpointer:
    def __init__(
        self,
        model:torch.nn.Module,
        thresholds:Tuple[float, float]
        ) -> None:

        self.model = model
        self.thresholds = thresholds

        self.checkpoint_path = os.path.join(os.getcwd(), "ckpt")
        self.check_folder()

        self.last_state_path = None

    def check_folder(self):
        """
        Creating ckpt folder if doesn't exists
        """

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
            "thresholds": self.thresholds,
            "description": desc,
        }

        torch.save(state_dict, self.last_state_path)
