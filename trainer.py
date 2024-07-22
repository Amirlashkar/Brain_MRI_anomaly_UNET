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

    def _get_train_csv(self) -> pd.DataFrame:
        """
        Returns train csv
        """

        train_csv_path = os.path.join(self.data_path, "train.csv")
        train_csv = pd.read_csv(train_csv_path)
        return train_csv
