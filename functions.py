from pydicom import dcmread, FileDataset
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from constants import *
import numpy as np
import os


data_path = os.path.join(os.getcwd(), "data", "main", "iaaa-mri-challenge", "data")

def read_dc(path: os.PathLike) -> FileDataset:
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
