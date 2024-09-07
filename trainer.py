from constants import *
import functions, utils, models
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from time import time


class Trainer:
    def __init__(self, chosen_protocol:str, do_preprocess:bool) -> None:
        # description for storing on checkpoints
        self.desc = str(input("Please write a description for your trainer:\n"))

        # making train.csv ready for preprocess
        self.data_path = os.path.join(os.getcwd(), "data", "main", "iaaa-mri-challenge") # <-------- !!!! change this line mentioning your data path
        self.train_csv = self._get_train_csv()
        self.detail_dict = self.detailing(self.train_csv)
        self.train_csv = self.filter_data(self.train_csv, chosen_protocol)
        self.normal_df, self.abnormal_df = self.separate_df(self.train_csv)

        # # in case of testing Trainer class faster
        # self.normal_df = self.normal_df.iloc[:20]
        # self.abnormal_df = self.abnormal_df.iloc[:10]

        self.last_state_path:Optional[str] = None

        # setting training device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Device: {self.device}")

        if do_preprocess:
            print("\nPreProcessing Phase\n--------------------")
            start = time()
            (
                self.train,
                self.val,
                self.val_labels,
            ) = self._get_train_val()
            end = time()
            print(f"PreProcess Time: {(end - start) / 60} Minutes")

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

    def _get_image_sp(self, patient_path:str) -> Tuple:
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
        of those patient data

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

    def filter_data(self, data:pd.DataFrame, chosen_protocol:str) -> pd.DataFrame:
        """
        Filters patients id df by detail_dict and with respect to chosen parameters

        data: patients id df
        chosen_shape: images shape to keep
        chosen_protocol: images protocol to keep
        """

        chosens = []
        for patient_id, dict_ in self.detail_dict.items():
            protocol = dict_["protocol"]
            if protocol == chosen_protocol:
                chosens.append(patient_id)

        data_ = data.copy()
        data_ = data_[data_["SeriesInstanceUID"].isin(chosens)]

        return data_

    def _get_train_val(self) -> Tuple:
        """
        Provides training & validation data
        """

        n_abnormal = self.abnormal_df.shape[0]
        normal_val = self.normal_df.iloc[-n_abnormal:]
        train_patients = self.normal_df[:-n_abnormal]

        # validation data consists of normal data on first half and abnormal data on the other half
        val_patients = pd.concat([normal_val, self.abnormal_df], axis=0)
        val_labels = val_patients["prediction"].to_numpy()

        train_images = []
        val_images = []
        for i, patient in enumerate(train_patients["SeriesInstanceUID"]):
            # doing validation preprocess on common iteration also
            if i < val_patients.shape[0]:
                val_patient_path = os.path.join(self.data_path, "data", val_patients.iloc[i]["SeriesInstanceUID"])

                patient_images = []
                for image in functions.iterate_patient(val_patient_path):
                    patient_images.append(image)

                if len(patient_images) == 0: # if sagittal
                    val_labels = np.delete(val_labels, i)
                else:
                    patient_images = np.array(patient_images, dtype=np.float32)
                    val_images.append(patient_images)

            # removing unideal data
            if patient in T1_RM_PATIENTS or patient in T2_RM_PATIENTS or patient in FLAIR_RM_PATIENTS:
                continue

            patient_path = os.path.join(self.data_path, "data", patient)
            for image in functions.iterate_patient(patient_path):
                # rot_image = functions.rotate(image).astype(np.float32)
                train_images.append(image) # adding both raw and rotated image

        train_images = np.array(train_images, dtype=np.float32)

        train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
        val_images = [torch.tensor(images, dtype=torch.float32).unsqueeze(1).to(self.device) for images in val_images]

        # # for the purpose of testing
        val_images = val_images[-50:]
        val_labels = val_labels[-50:]

        return train_images, val_images, val_labels

    def fit(self, *, pretrained_path:Optional[str]=None, freeze_model:Optional[bool]=None, epoch_val:bool) -> None:
        """
        Fits data into model to train

        pretrained_path: path to pretrained model checkpoint
        freeze_model: if you want to change two last layer learning
        epoch_val: does user want to validate model after each epoch or not
        """

        print("\nTraining Phase\n--------------------")

        # Giving caution if future computing load gonna be high
        if BATCH_SIZE > 16:
            answer = input(f"Chosen batch size is {BATCH_SIZE}! shall we continue? y/n")
            if str(answer) == "n":
                print("Training stopped!")
                return

        train_ds = utils.ImageDataset(self.train)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DL_WORKERS
        )

        # index of checkpoints on each epoch
        checkpoints = [int((i/5)*(len(train_dl)-1)) for i in range(1, 6)]
        model = models.UNet().to(self.device)

        # Partition for transfer learning from another checkpoint
        if pretrained_path:
            state, _, _ = self._load_state(self.device, pretrained_path)
            model.load_state_dict(state)
            # customize freezing layers with your needs
            if freeze_model:
                for param in model.enc1.parameters():
                    param.requires_grad = False
                for param in model.enc2.parameters():
                    param.requires_grad = False
                for param in model.enc3.parameters():
                    param.requires_grad = False
                for param in model.enc4.parameters():
                    param.requires_grad = False
                for param in model.bottleneck.parameters():
                    param.requires_grad = False
                for param in model.upconv4.parameters():
                    param.requires_grad = False
                for param in model.dec4.parameters():
                    param.requires_grad = False
                for param in model.upconv3.parameters():
                    param.requires_grad = False
                for param in model.dec3.parameters():
                    param.requires_grad = False

        criterion = utils.PXLoss(self.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        anomaly_scores = []
        for epoch in range(N_EPOCHS):
            print(f"\nEPOCH: {epoch+1}")
            losses = []
            batch_num = 0
            for batch in train_dl:
                batch = batch.to(self.device)

                # noising data before feeding it to data
                noised = functions.noise_batch(batch)
                reconstructs = model(noised)

                loss = criterion(reconstructs, batch)
                losses.append(loss)

                # partition for determining normal anomaly score threshold so we can compare test data anomaly scores with it
                with torch.no_grad():
                    # cloning current state model
                    anomaly_model = models.UNet().to(self.device)
                    anomaly_model.load_state_dict(model.state_dict())
                    a_recon = anomaly_model(batch)
                    anomaly_score, _ = functions.anomaly_scoring(a_recon, batch)
                    anomaly_scores.append(anomaly_score)

                    # Saving checkpoints
                    if batch_num in checkpoints:
                        print(f"----\n**Checkpint {checkpoints.index(batch_num)+1}")
                        anomaly_scores_ = torch.tensor(anomaly_scores)
                        thresholds = (torch.mean(anomaly_scores_).item(), torch.std(anomaly_scores_).item())
                        print(f"Thresolds: {thresholds}")

                        ckp = utils.Checkpointer(model, thresholds)
                        losses_ = torch.stack(losses)
                        avg_loss = torch.mean(losses_, dim=0).item()

                        ckp.save(epoch+1, avg_loss, self.desc)
                        self.last_state_path = ckp.last_state_path
                        print("State saved!\n----")
                        anomaly_scores = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Batch {batch_num+1} of {len(train_dl)}|Loss: {'{:.6f}'.format(loss.item())}")
                batch_num += 1

            losses = torch.stack(losses)
            avg_loss = torch.mean(losses, dim=0)
            print(f"Avg Epoch Loss: {avg_loss}")

            if epoch_val:
                self.inferences(self.last_state_path)

    def _load_state(self, device:str, path:Optional[str]=None) -> Tuple:
        """
        Loads state dict from provided path or last saved state if no path provided

        device: device to store model state inside (in case of not hitting torch future version error)
        path: wanted state dict path
        """

        if path:
            state_dict = torch.load(path, map_location=device, weights_only=False)
        else:
            state_dict = torch.load(self.last_state_path, map_location=device, weights_only=False)

        model_state = state_dict["model_state"]
        thresholds = state_dict["thresholds"]
        description = state_dict["description"]

        return model_state, thresholds, description

    def inferences(self, pretrained_path:Optional[str]=None, do_plot=False) -> None:
        """
        Calculates inferences of model on validation data

        pretrained_path: path of ckpt for testing otherwise function will use last made checkpoint on fit function
        """

        print("\nValidation Phase\n--------------------")

        model_state , thresholds, _ = self._load_state(self.device, pretrained_path)
        model = models.UNet().to(self.device)
        model.load_state_dict(model_state)

        predictions = functions.predict(model, self.val, thresholds, do_plot)
        c_matrix = confusion_matrix(self.val_labels, predictions)
        functions.print_pr_auc(c_matrix[0][0], c_matrix[0][1], c_matrix[1][0], c_matrix[1][1])
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)

        if self.last_state_path:
            cm_dir = os.path.dirname(self.last_state_path)
        else:
            cm_dir = os.getcwd()

        cm_path = os.path.join(cm_dir, "CM.png")
        plt.savefig(cm_path)

if __name__ == "__main__":
    protocols = [
        "T1W_SE",
        "T2W_FLAIR",
        "T2W_TSE"
    ]
    chosen_p = protocols[0]

    trainer = Trainer(chosen_p, True)
    print(f"Normal samples: {trainer.normal_df.shape[0]} | Abnormal samples: {trainer.abnormal_df.shape[0]}")
    # trainer.fit(epoch_val=True)
    trainer.inferences("/Users/albk/Documents/Code/Hackathons/IAAA/ckpt/best/alpha.5_T1_score.75_VA/state.pth", True)
