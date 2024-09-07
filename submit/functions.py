from pydicom import dcmread, FileDataset
from typing import List, Tuple, Generator
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from constants import *
import numpy as np
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _quadruple
import torch
import os, cv2

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
    Smoothes image that helps masking be more accurate on brain or skull edges.

    image: input image
    """

    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    smoothed = cv2.GaussianBlur(image, (7, 7), 0)
    smoothed = smoothed.astype(np.float32)
    return smoothed

def masking(image:np.ndarray):
    """
    Gets mask out of images that only include brain areas

    image: brain image
    """

    image = img_smoothing(image)
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    mask = image > .01
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = mask.astype(np.int8)

    return mask

def resize(image:np.ndarray) -> np.ndarray:
    """
    Resizes image to wanted shape on constants.py file

    image: image to resize
    """

    resized_image = cv2.resize(image, SHAPE, interpolation=cv2.INTER_CUBIC)
    return resized_image.astype(np.float32)

def iterate_patient(patient_path:str) -> Generator:
    """
    Provides image arrays with respect to provided patient path

    patient_path: path of images
    """

    volume = os.listdir(patient_path)
    try:
        volume.remove(".DS_Store")
    except:
        pass

    for image in volume:
        image_path = os.path.join(patient_path, image)
        ds = read_dc(image_path)
        protocol = str(ds["SeriesDescription"].value)
        slice_loc = float(ds["SliceLocation"].value)
        slice_ori = str(ds["2001", "100b"].value)
        if not SLICE_START < slice_loc < SLICE_END or slice_ori != "TRANSVERSAL": # filtering neck and vertex-close slices && aslo filtering non-axial slices
            continue

        image_arr = ds.pixel_array
        image_arr = resize(image_arr)
        image_arr = cropping(image_arr)
        image_arr = image_arr.astype(np.float32)
        image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())
        if protocol == "T2W_TSE":
            mask = masking(image_arr)
            image_arr = 1 - image_arr
            image_arr *= mask.astype(np.int8)

        yield image_arr

def median_pool(x, kernel_size=3, stride=1, padding=0):
    """
    Makes anomalous areas more visible

    x: SSIM map
    """

    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)

    x = F.pad(x, padding, mode='reflect')
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

    return x

def _patching(residual:torch.Tensor) -> Tuple[list, list]:
    """
    Creates patches out of an image.

    residual: SSIM map
    """

    residual = residual.squeeze(0).squeeze(0).cpu().detach().numpy()

    patches = []
    coors = []
    for i in range(0, SHAPE[0], PATCH_SKIP):
        for j in range(0, SHAPE[0], PATCH_SKIP):
            patch = residual[i:i+PATCH_DIM, j:j+PATCH_DIM]
            patches.append(patch)
            coors.append((j, i))

    return patches, coors

def high_patch_criterion(arr:np.ndarray) -> float:
    """
    Calculates a criteria that image patches should be sorted by it on _high_patch func.

    arr: array of patch
    """

    arr = np.nan_to_num(arr, nan=0)
    threshold = np.percentile(arr, 80)
    top_quarter = arr[arr >= threshold]
    mean = np.mean(top_quarter).item()

    if np.isnan(mean):
        return 0
    else:
        return mean

def top_part_mean(arr:np.ndarray) -> float:
    """
    Gets mean of outlier data within a patch or a sample array.

    arr: input array to get outlier mean of it
    """

    arr = np.nan_to_num(arr, nan=0)
    threshold = np.mean(arr) + np.std(arr)
    top_quarter = arr[arr >= threshold]
    mean = np.mean(top_quarter).item()

    if np.isnan(mean):
        return 0
    else:
        return mean

def _high_patch(patches:list, coors:list) -> Tuple:
    """
    Hunts down the most anomalous patch within an image.

    patches: all patches of that image
    coors: coordination of patches ; same as patches ordination
    """

    patches_mean = np.array([high_patch_criterion(patch.flatten()) for patch in patches])
    high_mean = np.sort(patches_mean)[-1]
    ind = patches_mean.tolist().index(high_mean)
    high_patch = patches[ind]
    chosen_coor = coors[ind]
    return high_patch, chosen_coor

def batch_ssim(predictions, targets):
    """
    Calculates SSIM map of predictions and input of model ;
    This map tells us where exactly two images are different from eachother.

    predictions: batch of model predictions out of model input batch; SHAPE:(n, 1, 256, 256)
    targets: batch of model input; SHAPE:(n, 1, 256, 256)
    """

    diff = []
    for i, img in enumerate(predictions):
        p_img = img[0].cpu().detach().numpy()
        t_img = targets[i][0].cpu().detach().numpy()
        mask = masking(t_img)

        p_img = (p_img - p_img.min()) / (p_img.max() - p_img.min()) * 255
        p_img = p_img.astype(np.uint8)
        t_img = (t_img - t_img.min()) / (t_img.max() - t_img.min()) * 255
        t_img = t_img.astype(np.uint8)

        _, ssim_img = ssim(t_img, p_img, full=True, data_range=1.)
        ssim_img = 1 - ssim_img # we want differences
        ssim_img *= mask
        ssim_img += 1
        ssim_img = torch.tensor(ssim_img, dtype=torch.float32)
        diff.append(ssim_img)

    diff = torch.stack(diff).unsqueeze(1)

    return diff

def anomaly_scoring(predictions:torch.Tensor, targets:torch.Tensor) -> Tuple:
    """
    Compares two batch of images and give them an score of how much the targets batch
    is anomalous.

    predictions: batch of model predictions out of model input batch; SHAPE:(n, 1, 256, 256)
    targets: batch of model input; SHAPE:(n, 1, 256, 256)
    """

    ssim = batch_ssim(predictions, targets)

    # suppressing low values
    residual = torch.pow(ssim, 2).to(targets.device)
    residual = median_pool(residual, kernel_size=5, stride=1, padding=2)

    residual = residual.squeeze(1)
    high_p_scores = []
    for r in residual:
        ps, cs = _patching(r)
        high_p, _ = _high_patch(ps, cs)
        patch_var = np.var(high_p.flatten())
        top_mean = top_part_mean(high_p.flatten())
        score = top_mean + patch_var
        high_p_scores.append(score)

    high_p_scores = np.sort(np.array(high_p_scores))[-2:]
    anomaly_score = np.mean(high_p_scores).item()

    return anomaly_score, residual

def predict(
        model:torch.nn.Module,
        patients_image:List[torch.Tensor],
        thresholds:Tuple,
    ) -> np.ndarray:
    """
    Provides model-driven labels (predictions)

    model: a model to predict batch with
    patients_image: list of all patients images (each patient may have different count of images)
    thresholds: avg and std of training phase pixel-wise model loss
    """

    predictions = []
    for patient in patients_image:
        with torch.no_grad():
            reconstructs = model(patient)
            anomaly_score, _ = anomaly_scoring(reconstructs, patient)
            avg, std = thresholds
            thres = avg + .5*std

            if anomaly_score > thres:
                predictions.append(1)
            else:
                predictions.append(0)

    return np.array(predictions)
