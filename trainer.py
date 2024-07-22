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

        self.train:Optional[np.ndarray] = None
        self.val:Optional[np.ndarray] = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def _save_model(self, model:components.AutoEncoder) -> None:
        """
        Saves model on specific path

        model: model to be saved
        """

        saving_path = os.path.join(os.getcwd(), "model.pth")
        torch.save(model.state_dict(), saving_path)

    def compare_images(self, orig_batch:torch.Tensor, recon_batch:torch.Tensor) -> Tuple:
        """
        Compares two set of images by numerical criterias

        orig_batch: original images
        recon_batch: reconstructed images
        """

        orig_batch = orig_batch.detach().numpy()
        recon_batch = recon_batch.detach().numpy()

        mse_ls = []
        ssim_ls = []
        nrmse_ls = []
        cc_ls = []
        for i, image in enumerate(orig_batch):
            mse = mean_squared_error(image, recon_batch[i])
            data_range = image.max() - image.min()
            ssim, _ = structural_similarity(np.squeeze(image), np.squeeze(recon_batch[i]), full=True, data_range=data_range)
            nrmse = np.sqrt(mse) / (image.max() - image.min())
            cc = np.corrcoef(image.flatten(), recon_batch[i].flatten())[0, 1]

            mse_ls.append(mse)
            ssim_ls.append(ssim)
            nrmse_ls.append(nrmse)
            cc_ls.append(cc)

        return mse_ls, ssim_ls, nrmse_ls, cc_ls

    def save_thresholds(self, mse:list, ssim:list, nrmse:list, cc:list) -> None:
        """
        Calculating comparision threshold based on inputs avg and standard deviation

        mse: list of images mse
        ssim: list of images ssim
        nrmse: list of images nrmse
        cc: list of images cc
        """

        mse = np.array(mse)
        ssim = np.array(ssim)
        nrmse = np.array(nrmse)
        cc = np.array(cc)

        mse_avg = np.mean(mse)
        ssim_avg = np.mean(ssim)
        nrmse_avg = np.mean(nrmse)
        cc_avg = np.mean(cc)

        mse_std = np.std(mse)
        ssim_std = np.std(ssim)
        nrmse_std = np.std(nrmse)
        cc_std = np.std(cc)

        thresholds_path = os.path.join(os.getcwd(), "thresholds.pkl")
        with open(thresholds_path, "wb") as file:
            pickle.dump((
                mse_avg, mse_std,
                ssim_avg, ssim_std,
                nrmse_avg, nrmse_std,
                cc_avg, cc_std,
            ), file)

    def fit(self) -> None:
        """
        Fits data into model to train
        """

        train_images, _, _ = self._get_train_val()
        train_ds = components.ImageDataset(train_images)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        model = components.AutoEncoder().to(self.device)
        criterion = nn.MSELoss()
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

        self.save_thresholds(mse_, ssim_, nrmse_, cc_)
        self._save_model(model)

    def _load_thresholds(self) -> Tuple:
        """
        Loads Avg and Std on each criteria from file
        """

        thresholds_path = os.path.join(os.getcwd(), "thresholds.pkl")
        with open(thresholds_path, "rb") as file:
            (mse_avg, mse_std,
            ssim_avg, ssim_std,
            nrmse_avg, nrmse_std,
            cc_avg, cc_std) = pickle.load(file)

        return (mse_avg, mse_std,
            ssim_avg, ssim_std,
            nrmse_avg, nrmse_std,
            cc_avg, cc_std)

    def predict(self, model:components.AutoEncoder, patients_image:List[np.ndarray], criterias:Tuple) -> np.ndarray:
        main_predictions = []
        for i, patient in enumerate(patients_image):
            print(f"{i} from {len(patients_image)}")
            reconstructs = model(patient)
            mse, ssim, nrmse, cc = self.compare_images(patient, reconstructs)

            (mse_avg, mse_std,
            ssim_avg, ssim_std,
            nrmse_avg, nrmse_std,
            cc_avg, cc_std) = criterias

            anomaly_degrees = []
            for i in range(len(mse)):
                mse_ = mse[i]
                ssim_ = ssim[i]
                nrmse_ = nrmse[i]
                cc_ = cc[i]

                anomaly_degree = 0

                if not (mse_avg - mse_std < mse_ < mse_avg + mse_std):
                    print("mse")
                    anomaly_degree =+ 1

                if not (ssim_avg - ssim_std < ssim_ < ssim_avg + ssim_std):
                    anomaly_degree =+ 1

                if not (nrmse_avg - nrmse_std < nrmse_ < nrmse_avg + nrmse_std):
                    print("nrmse")
                    anomaly_degree =+ 1

                if not (cc_avg - cc_std < cc_ < cc_avg + cc_std):
                    print("cc")
                    anomaly_degree =+ 1

                anomaly_degrees.append(anomaly_degree)

                first_predictions = []
                for degree in anomaly_degrees:
                    if degree > 0:
                        first_predictions.append(1)
                    else:
                        first_predictions.append(0)

            second_prediction = np.array(first_predictions, dtype=np.int8).sum()
            if second_prediction >= ANOMALY_LIMIT:
                main_predictions.append(1)
            else:
                main_predictions.append(0)

        return np.array(main_predictions, dtype=np.int8)

    def inferences(self):
        """
        Calculates inferences of model on validation data
        """

        model_path = os.path.join(os.getcwd(), "model.pth")
        model = components.AutoEncoder().to(self.device)
        model.load_state_dict(torch.load(model_path))

        _, val_images, val_labels = self._get_train_val()
        criterias = self._load_thresholds()

        predictions = self.predict(model, val_images, criterias)
        c_matrix = confusion_matrix(val_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("CM.png")
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
