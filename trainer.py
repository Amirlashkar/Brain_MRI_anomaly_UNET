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

    def multi_plot(self, ids:List[str]) -> None:
        """
        *** Analytic Function ***
        Plots slices of multiple patients

        ids: list of patient ids to plot their data
        """

        max_slices = 0
        multi_slices = []
        for id in ids:
            patient_path = os.path.join(self.data_path, "data", id)
            slices = os.listdir(patient_path)
            try:
                slices.remove(".DS_Store")
            except:
                pass

            slices_count = len(slices)
            max_slices = max_slices if slices_count < max_slices else slices_count

            temp_ls = []
            for slice in slices:
                slice_path = os.path.join(patient_path, slice)
                img = functions.read_dc(slice_path).pixel_array
                temp_ls.append(img)

            multi_slices.append(temp_ls)

        fig, axs = plt.subplots(len(ids), max_slices, figsize=(15, 5))
        for i, patient in enumerate(multi_slices):
            for ind, slice in enumerate(patient):
                axs[i, ind].imshow(slice, aspect="auto", cmap="gray")
                axs[i, ind].axis('off')

        plt.tight_layout()
        plt.show()

    def _iterate_images(self, patient_path:os.PathLike) -> Generator:
        """
        Provides image arrays with respect to provided path

        patient_path: path of images
        """

        images = os.listdir(patient_path)
        try:
            images.remove(".DS_Store")
        except:
            pass

        for image in images:
            image_path = os.path.join(patient_path, image)
            image_arr = functions.read_dc(image_path).pixel_array
            image_arr = np.expand_dims(image_arr, axis=0) # adding single channel to each image
            yield image_arr

    def _save_scaler(self, scaler:StandardScaler) -> None:
        """
        Saves scaler as pickle file

        scaler: scaler to be saved
        """

        scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
        with open(scaler_path, "wb") as file:
            pickle.dump(scaler, file)

    def _get_train_val(self) -> Tuple[torch.Tensor, List[torch.Tensor], np.ndarray]:
        """
        Provides training & val data
        """

        n_abnormal = self.abnormal_df.shape[0]
        normal_val = self.normal_df.iloc[-n_abnormal:]
        train_patients = self.normal_df[:-n_abnormal]
        val_patients = pd.concat([normal_val, self.abnormal_df], axis=0)
        val_labels = val_patients["prediction"].to_numpy()

        train_images = []
        val_images = []
        for i, patient in enumerate(train_patients["SeriesInstanceUID"]):
            if i < val_patients.shape[0]:
                val_patient_path = os.path.join(self.data_path, "data", val_patients.iloc[i]["SeriesInstanceUID"])
                patient_images = []
                for image in self._iterate_images(val_patient_path):
                    patient_images.append(image)

                patient_images = np.array(patient_images, dtype=np.float32)
                val_images.append(patient_images)

            patient_path = os.path.join(self.data_path, "data", patient)
            for image in self._iterate_images(patient_path):
                train_images.append(image)

        train_images = np.array(train_images, dtype=np.float32)

        train_images, scaler = functions.data_scale(train_images)
        self._save_scaler(scaler)
        val_images = [
            functions.data_scale(arr, scaler) for arr in val_images
        ]

        train_images = torch.tensor(train_images, dtype=torch.float32)
        val_images = [torch.tensor(images, dtype=torch.float32) for images in val_images]
        # val_images = val_images[200:212]
        # val_labels = val_labels[200:212]

        return train_images, val_images, val_labels

