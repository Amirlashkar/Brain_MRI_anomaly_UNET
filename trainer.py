from collections.abc import Generator
from constants import *
import functions, components
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, cv2
from sklearn.preprocessing import StandardScaler
from skimage.metrics import mean_squared_error, structural_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Trainer:
    def __init__(self, chosen_shapes:List[Tuple], chosen_protocols:List[str]) -> None:
        self.data_path = os.path.join(os.getcwd(), "data", "main", "iaaa-mri-challenge")
        self.train_csv = self._get_train_csv()
        self.detail_dict = self.detailing(self.train_csv)

    def _get_train_csv(self) -> pd.DataFrame:
        """
        Returns train csv
        """

        train_csv_path = os.path.join(self.data_path, "train.csv")
        train_csv = pd.read_csv(train_csv_path)
        return train_csv
    def detailing(self, data:pd.DataFrame) -> dict:
        """
        Creates a dictionary of patient ids on keys and values of another dictionary carrying some detail

        data: data with patients id
        """

        detail_dict = {}
        patients = data["SeriesInstanceUID"]
        for patient in patients:
            # label
            prediction = data[data["SeriesInstanceUID"] == patient]["prediction"]

            # image shape & protocol
            patient_path = os.path.join(self.data_path, "data", patient)
            shape, protocol = self._get_image_sp(patient_path)

            # slice count
            slices = os.listdir(patient_path)
            try:
                slices.remove(".DS_Store")
            except:
                pass

            slice_count = len(slices)

            # inserting details with key of patient id
            detail_dict[patient] = {
                "label": prediction,
                "shape": shape,
                "protocol": protocol,
                "slice_count": slice_count
            }

        return detail_dict

