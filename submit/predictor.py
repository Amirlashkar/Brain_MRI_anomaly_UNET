import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Tuple
from models import UNet
import functions


class Predictor:
    def __init__(self, data_dir:Path) -> None:
        self.data_dir = data_dir
        self.ckpt_dir = os.path.join(os.getcwd(), "ckpt")

        # setting training device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        (self.t1_model, self.t1_thres,
        self.t2_model, self.t2_thres,
        self.flair_model, self.flair_thres) = self._create3modmodel()

    def _load_state(self, path:str) -> Tuple:
        """
        Loads state dict from provided path or last saved state if no path provided

        path: wanted state dict path
        """

        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        model_state = state_dict["model_state"]
        thresholds = state_dict["thresholds"]
        description = state_dict["description"]

        return model_state, thresholds, description

    def _create3modmodel(self) -> Tuple:
        """
        Creates three model for three modalities that'll be used whenever its needed.
        """

        t1_state, t1_thres, _ = self._load_state(os.path.join(self.ckpt_dir, "t1.pth"))
        t2_state, t2_thres, _ = self._load_state(os.path.join(self.ckpt_dir, "t2.pth"))
        flair_state, flair_thres, _ = self._load_state(os.path.join(self.ckpt_dir, "flair.pth"))

        t1_model = UNet().to(self.device)
        t2_model = UNet().to(self.device)
        flair_model = UNet().to(self.device)
        t1_model.load_state_dict(t1_state)
        t2_model.load_state_dict(t2_state)
        flair_model.load_state_dict(flair_state)

        return (t1_model, t1_thres,
                t2_model, t2_thres,
                flair_model, flair_thres)

    def predict(self) -> pd.DataFrame:
        """
        Predicts series on data directory and gives back the predictions with respect to the series uid
        and returns dataframe of predictions.
        """

        series_uid = os.listdir(self.data_dir)

        predictions = []
        for uid in series_uid:
            uid_path = os.path.join(self.data_dir, uid)
            images_path = os.listdir(uid_path)

            sample_image_path = os.path.join(uid_path, images_path[0])
            sample = functions.read_dc(sample_image_path)
            protocol = str(sample["SeriesDescription"].value)

            if "FLAIR" in protocol:
                model = self.flair_model
                thres = self.flair_thres
            elif "T2" in protocol:
                model = self.t2_model
                thres = self.t2_thres
            else:
                model = self.t1_model
                thres = self.t1_thres

            images = []
            for img in functions.iterate_patient(uid_path):
                images.append(img)

            images = np.array(images, dtype=np.float32)
            images = [torch.tensor(images, dtype=torch.float32).unsqueeze(1).to(self.device)]
            prediction = functions.predict(model, images, thres)[0]
            predictions.append(prediction)

        out_df = pd.DataFrame(
            {
                "SeriesInstanceUID": series_uid,
                "prediction": predictions
            }
        )

        return out_df
