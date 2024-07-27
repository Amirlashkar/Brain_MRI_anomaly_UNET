from collections.abc import Generator
from constants import *
import functions, utils, models
from typing import List, Optional, Tuple
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Trainer:
    def __init__(self, chosen_shapes:List[Tuple], chosen_protocols:List[str]) -> None:
        self.desc = str(input("Please write a description for your trainer:\n"))

        self.data_path = os.path.join(os.getcwd(), "data", "main", "iaaa-mri-challenge")
        self.train_csv = self._get_train_csv()
        self.detail_dict = self.detailing(self.train_csv)
        self.train_csv = self.filter_data(self.train_csv, chosen_shapes, chosen_protocols)
        self.normal_df, self.abnormal_df = self.separate_df(self.train_csv)

        self.train:Optional[np.ndarray] = None
        self.val:Optional[np.ndarray] = None
        self.scaler:Optional[StandardScaler] = None
        self.last_state_path:Optional[str] = None

        # setting training device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        else:
            self.device = torch.device("cpu")

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

    def _get_image_sp(self, patient_path:os.PathLike) -> Tuple:
        """
        Provides image shape and used protocol from patient_path

        patient_path: path of patient images
        """

        files = os.listdir(patient_path)
        try:
            files.remove(".DS_Store")
        except:
            pass

        sample = files[0]
        sample_path = os.path.join(patient_path, sample)
        ds = functions.read_dc(sample_path)
        shape = ds.pixel_array.shape
        protocol = ds.SeriesDescription

        return shape, protocol

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
                for image in functions.iterate_patient(val_patient_path):
                    patient_images.append(image)

                patient_images = np.array(patient_images, dtype=np.float32)
                val_images.append(patient_images)

            patient_path = os.path.join(self.data_path, "data", patient)
            for image in functions.iterate_patient(patient_path):
                train_images.append(image)

        train_images = np.array(train_images, dtype=np.float32)

        train_images, self.scaler = functions.data_scale(train_images)
        val_images = [
            functions.data_scale(arr, self.scaler) for arr in val_images
        ]

        train_images = torch.tensor(train_images, dtype=torch.float32)
        # val_images = val_images[200:212]
        # val_labels = val_labels[200:212]
        val_images = [torch.tensor(images, dtype=torch.float32).to(self.device) for images in val_images]

        return train_images, val_images, val_labels

    def fit(self) -> None:
        """
        Fits data into model to train
        """

        train_images, _, _ = self._get_train_val()
        train_ds = components.ImageDataset(train_images)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        model = components.AutoEncoder().to(self.device)
        criterion = nn.MSELoss()
        checkpoints = [int((i/5)*len(train_dl)) for i in range(1, 6)]
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        mse_ = []
        ssim_ = []
        nrmse_ = []
        cc_ = []
        for epoch in range(N_EPOCHS):
            print(f"EPOCH: {epoch+1}")
            losses = []
            for batch in train_dl:
                batch = batch.to(self.device)
                reconstructs = model(batch)
                loss = criterion(reconstructs, batch)
                losses.append(loss)
                if epoch > 0:
                    anomaly_scores.append(loss.item())
                    if i in checkpoints:
                        print(f"**Checkpint {checkpoints.index(i)}")
                        anomaly_scores_ = torch.tensor(anomaly_scores)
                        thresholds = (torch.mean(anomaly_scores_).item(), torch.std(anomaly_scores_).item())

                        ckp = utils.Checkpointer(model, self.scaler, thresholds)
                        losses_ = torch.stack(losses)
                        avg_loss = torch.mean(losses_, dim=0).item()

                        ckp.save(epoch+1, avg_loss, self.desc)
                        self.last_state_path = ckp.last_state_path
                        print("State saved!")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # calculating comparing criterias
                mse, ssim, nrmse, cc = self.compare_images(batch, reconstructs)
                mse_.extend(mse)
                ssim_.extend(ssim)
                nrmse_.extend(nrmse)
                cc_.extend(cc)

                print(f"Loss: {loss}")

            losses = torch.stack(losses)
            avg_loss = torch.mean(losses, dim=0)
            print(f"Avg Epoch Loss: {avg_loss}")

    def _load_state(self, device:DeviceLikeType, path:Optional=None) -> Tuple:
        """
        Loads state dict from provided path or last saved state if no path provided

        path: wanted state dict path
        """

        if path:
            state_dict = torch.load(path, map_location=device)
        else:
            state_dict = torch.load(self.last_state_path, map_location=device)

        model_state = state_dict["model_state"]
        scaler = state_dict["scaler"]
        thresholds = state_dict["thresholds"]
        description = state_dict["description"]

        return model_state, scaler, thresholds, description

    def inferences(self, pretrained_model_path:Optional[str]=None) -> None:
        """
        Calculates inferences of model on validation data
        """
        print("\nValidation Phase\n--------------------")

        model_state , scaler, thresholds, _ = self._load_state(self.device, pretrained_model_path)
        model = models.UNet().to(self.device)
        model.load_state_dict(model_state)

        _, val_images, val_labels = self._get_train_val()

        criterion = utils.PXLoss(self.device)
        predictions = functions.predict(model, val_images, thresholds, criterion, scaler)
        c_matrix = confusion_matrix(val_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)

        if self.last_state_path:
            cm_dir = os.path.dirname(self.last_state_path)
        else:
            cm_dir = os.getcwd()

        cm_path = os.path.join(cm_dir, "CM.png")
        plt.savefig(cm_path)

if __name__ == "__main__":
    shapes = [
        (288, 288),
        # (256, 256),
    ]
    protocols = [
        "T1W_SE",
        "T2W_FLAIR",
        # "T2W_TSE"
    ]

    trainer = Trainer(shapes, protocols)
    print(trainer.normal_df.shape, trainer.abnormal_df.shape)
