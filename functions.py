from pydicom import dcmread, FileDataset
from typing import List, Tuple, Generator
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from constants import *
import numpy as np
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _quadruple
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

def crop(vol, mins, maxs):
    """Crops the input volume.

    Parameters
    ----------
    vol : ndarray
        Volume to crop.
    mins : array
        Array containing minimum index of each dimension.
    maxs : array
        Array containing maximum index of each dimension.

    Returns
    -------
    vol : ndarray
        The cropped volume.
    """
    return vol[tuple(slice(i, j) for i, j in zip(mins, maxs))]

def bounding_box(vol):
    """Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.

    Returns
    -------
    npmins : list
        Array containing minimum index of each dimension
    npmaxs : list
        Array containing maximum index of each dimension
    """
    # Find bounds on first dimension
    temp = vol
    for i in range(vol.ndim - 1):
        temp = temp.any(-1)
    mins = [temp.argmax()]
    maxs = [len(temp) - temp[::-1].argmax()]
    # Check that vol is not all 0
    if mins[0] == 0 and temp[0] == 0:
        warn('No data found in volume to bound. Returning empty bounding box.')
        return [0] * vol.ndim, [0] * vol.ndim
    # Find bounds on remaining dimensions
    if vol.ndim > 1:
        a, b = bounding_box(vol.any(0))
        mins.extend(a)
        maxs.extend(b)
    return mins, maxs

def applymask(vol, mask):
    """ Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Array with $V$ dimensions
    mask : ndarray
        Binary mask.  Has $M$ dimensions where $M <= V$. When $M < V$, we
        append $V - M$ dimensions with axis length 1 to `mask` so that `mask`
        will broadcast against `vol`.  In the typical case `vol` can be 4D,
        `mask` can be 3D, and we append a 1 to the mask shape which (via numpy
        broadcasting) has the effect of applying the 3D mask to each 3D slice in
        `vol` (``vol[..., 0]`` to ``vol[..., -1``).

    Returns
    -------
    masked_vol : ndarray
        `vol` multiplied by `mask` where `mask` may have been extended to match
        extra dimensions in `vol`
    """
    mask = mask.reshape(mask.shape + (vol.ndim - mask.ndim) * (1,))
    return vol * mask

def otsu(image, nbins=256):
    """
    Return threshold value based on Otsu's method.
    Copied from scikit-image to remove dependency.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    hist, bin_centers = np.histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers[1:]) / weight1
    mean2 = (np.cumsum((hist * bin_centers[1:])[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def cropping(image:np.ndarray) -> np.ndarray:
    """
    Create a mask to only conclude most valuable regions of brain

    image: image or batch of images to create mask from it
    """

    thres = otsu(image)
    mask = image > thres
    mask = morphology.remove_small_holes(mask, area_threshold=20000)
    mins, maxs = bounding_box(mask)
    mask = crop(mask, mins, maxs)
    image = crop(image, mins, maxs)
    image += 1 # letting model see zero values also
    masked_image = applymask(image, mask)
    masked_image = resize(masked_image)

    return masked_image

def img_smoothing(image):
    """
    This function is used on masking to make mask edges clearer

    image: grayscale image of brain
    """

    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    smoothed = cv2.GaussianBlur(image, (7, 7), 0)
    smoothed = smoothed.astype(np.float32)
    return smoothed

def masking(image:np.ndarray):
    """
    Making mask which only includes brain areas

    image: brain grayscale image
    """

    image = img_smoothing(image)
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    mask = image > .01
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = mask.astype(np.int8)

    return mask

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

    mimg = image.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask = masking(mimg)
    mask = torch.tensor(mask).to(image.device)
    ns *= mask.unsqueeze(0).unsqueeze(0)

    image = image + ns

    return image.squeeze(0) # shape=(1, *SHAPE) (this is for preparing tensor for torch.stack)

def iterate_patient(patient_path:str) -> Generator:
def noise_batch(batch:torch.Tensor):
    """
    Provides image arrays with respect to provided patient path
    Adds noise to each element of a batch

    patient_path: path of images
    batch: batch to add noise to it
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
    noised = torch.stack([noise(image.unsqueeze(0), NOISE_RES, NOISE_STD) for image in batch]) # adding noise to raw image
    return noised.to(batch.device)

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
