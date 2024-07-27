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

def segment_brain(image:np.ndarray) -> np.ndarray:
    """
    Create a mask to only conclude most valuable regions of brain

    image: image or batch of images to create mask from it
    """

    threshold_value = filters.threshold_otsu(image)
    brain_mask = image > threshold_value

    # brain_mask = morphology.remove_small_objects(brain_mask, min_size=64)
    # brain_mask = morphology.remove_small_holes(brain_mask, area_threshold=64)

    return brain_mask

def iterate_patient(patient_path:str) -> Generator:
    """
    Provides image arrays with respect to provided patient path

    patient_path: path of images
    """

    images = os.listdir(patient_path)
    try:
        images.remove(".DS_Store")
    except:
        pass

    for image in images:
        image_path = os.path.join(patient_path, image)
        image_arr = read_dc(image_path).pixel_array
        image_arr = np.expand_dims(image_arr, axis=0) # adding single channel to each image
        yield image_arr

def plot_org_recon(org, recon):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(org, aspect="auto", cmap="gray")
    axs[0].axis('off')
    axs[0].set_title("Original")
    axs[1].imshow(recon, aspect="auto", cmap="gray")
    axs[1].axis('off')
    axs[1].set_title("Recon")

    plt.tight_layout()
    plt.show()

