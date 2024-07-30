from pydicom import dcmread, FileDataset
from typing import List, Optional, Tuple, Generator
from sklearn.preprocessing import StandardScaler
from skimage import filters, morphology
import matplotlib.pyplot as plt
from constants import *
import numpy as np
from torch.nn import functional as F
from torch._prims_common import DeviceLikeType
import torch
import os, cv2, random


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

def noise(image: torch.Tensor, noise_res:int, noise_std:float) -> torch.Tensor:
    """
    Adds noise to brain mask of given image

    image: given image (shape=(1, 1, *SHAPE))
    noise_res: initial noise resolution (bigger res mean bigger frequency of noise at image)
    noise_std: noise standard deviation
    """

    ns = torch.normal(mean=torch.zeros(image.shape[0], image.shape[1], noise_res, noise_res), std=noise_std).to(image.device)
    ns = F.upsample_bilinear(ns, size=[*SHAPE])

    roll_x = random.choice(range(SHAPE[0]))
    roll_y = random.choice(range(SHAPE[0]))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = segment_brain(image.cpu().detach().numpy())
    mask = torch.tensor(mask).to(image.device)
    ns *= mask

    image = image + ns

    return image.squeeze(0) # shape=(1, *SHAPE) (this is for preparing tensor for torch.stack)

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
        yield image_arr

def resize(image:np.ndarray) -> np.ndarray:
    """
    Resizes image to wanted shape on constants.py file

    image: image to resize
    """

    resized_image = cv2.resize(image, SHAPE, interpolation=cv2.INTER_CUBIC)
    return resized_image

def rotate(image:np.ndarray) -> np.ndarray:
    """
    Rotates given image

    image: image to rotate
    """

    center = (SHAPE[0] // 2, SHAPE[1] // 2)
    angle = random.randint(1, ROT_DEG)
    flip = bool(random.randint(0, 1))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (SHAPE[0], SHAPE[1]))
    if flip:
        rotated_image = cv2.flip(rotated_image, 1)

    return rotated_image

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

def predict(
        model:torch.nn.Module,
        patients_image:List[torch.Tensor],
        thresholds:Tuple,
        criterion:torch.nn.Module,
        scaler:StandardScaler
    ) -> np.ndarray:
    """
    Provides model-driven labels (predictions)

    model: a model to predict batch with
    patients_image: list of all patients images (each patient may have different count of images)
    thresholds: avg and std of training phase pixel-wise model loss
    """

    predictions = []
    for i, patient in enumerate(patients_image):
        print(f"{i+1} from {len(patients_image)}")
        reconstructs = model(patient)
        d_patient = data_descale(patient, scaler)
        d_reconstructs = data_descale(reconstructs, scaler)
        # raw = d_patient.cpu().detach().numpy()[0][0]
        # recon = d_reconstructs.cpu().detach().numpy()[0][0]
        # plot_org_recon(raw, recon)
        loss = criterion(d_reconstructs, d_patient).item()
        avg, std = thresholds

        if not (loss < avg + std):
            predictions.append(1)
        else:
            predictions.append(0)

        print(f"Loss: {loss} | Threshold: {avg + std}")

    return np.array(predictions, dtype=np.int8)
