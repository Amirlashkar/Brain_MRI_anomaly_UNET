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
        self.train_csv = self.filter_data(self.train_csv, chosen_shapes, chosen_protocols)
        self.normal_df, self.abnormal_df = self.separate_df(self.train_csv)

    def _get_train_csv(self) -> pd.DataFrame:
        """
        Returns train csv
        """

        train_csv_path = os.path.join(self.data_path, "train.csv")
        train_csv = pd.read_csv(train_csv_path)
        return train_csv

    def separate_df(self, train_csv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads train.csv then returns two dataframe of only normal and abnormal class separatly

        train_csv: data with patient ids
        """

        normal_df = train_csv[train_csv["prediction"] == 0]
        abnormal_df = train_csv[train_csv["prediction"] == 1]
        return normal_df, abnormal_df

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

    def filter_data(self, data:pd.DataFrame, chosen_shapes:List[Tuple], chosen_protocols:List[str]) -> pd.DataFrame:
        """
        Filters patients id df by detail_dict and with respect to chosen parameters

        data: patients id df
        chosen_shape: images shape to keep
        chosen_protocol: images protocol to keep
        """

        chosens = []
        for patient_id, dict_ in self.detail_dict.items():
            shape = dict_["shape"]
            protocol = dict_["protocol"]
            if shape in chosen_shapes and protocol in chosen_protocols:
                chosens.append(patient_id)

        data_ = data.copy()
        data_ = data_[data_["SeriesInstanceUID"].isin(chosens)]
        return data_

    def add_padding(self, image: np.ndarray, target_size=(288, 288)) -> np.ndarray:
        """
        Takes a 256*256 image and gives 288*288 image back

        image: 256*256 shape image
        """

        original_size = image.shape

        top = (target_size[0] - original_size[0]) // 2
        bottom = target_size[0] - original_size[0] - top
        left = (target_size[1] - original_size[1]) // 2
        right = target_size[1] - original_size[1] - left

        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return padded_image

