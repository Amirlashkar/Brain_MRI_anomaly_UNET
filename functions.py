from pydicom import dcmread
import numpy as np
import os


def read_dc(path: os.PathLike) -> np.ndarray:
    """
    This function reads array from a dicom file

    path: dicom file path
    """

    with open(path, "r") as file:
        ds = dcmread(file) # dicom complex content
        arr = ds.pixel_array

    return arr
