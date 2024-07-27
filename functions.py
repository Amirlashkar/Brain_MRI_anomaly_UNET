from pydicom import dcmread, FileDataset
from typing import List, Optional, Tuple, Generator
from sklearn.preprocessing import StandardScaler
from skimage import filters, morphology
import matplotlib.pyplot as plt
from constants import *
import utils
import numpy as np
from torch._prims_common import DeviceLikeType
import torch
import os


data_path = os.path.join(os.getcwd(), "data", "main", "iaaa-mri-challenge", "data")

def read_dc(path:str) -> FileDataset:
    """
    Reads dataset of a dicom file

    path: dicom file path
    """

    ds = dcmread(path) # dicom complex content

    return ds

def data_scale(data: np.ndarray, scaler:Optional[StandardScaler]=None) -> Tuple[np.ndarray, StandardScaler] | np.ndarray:
    """
    Scales data and gives it back ; if an scaler inserted then no scaler will be returned

    data: training data
    scaler: pre-made scaler
    """

    if not scaler:
        scaler_ = StandardScaler()

    n_samples = data.shape[0]
    data = data.reshape(n_samples * SHAPE[0], SHAPE[-1]) # preparing shape for scaler
    data = scaler_.fit_transform(data) if not scaler else scaler.transform(data)
    data = data.reshape(n_samples, 1, *SHAPE) # returning shape back to initial

    if not scaler:
        return data, scaler_
    else:
        return data

def data_descale(data: torch.Tensor, scaler:StandardScaler) -> torch.Tensor:
    """
    Converts image back to how it should be after taking scaler

    data: images to convert back
    scaler: fit scaler to use
    """

    n_samples = data.shape[0]
    data = data.reshape(n_samples * SHAPE[0], SHAPE[-1])
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=data.device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=data.device)
    data = (data * scale) + mean
    data = data.reshape(n_samples, 1, *SHAPE)

    return data
