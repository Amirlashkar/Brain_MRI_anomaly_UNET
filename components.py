import torch
import torch.nn as nn
from torch.utils.data import Dataset
from constants import *




class Encoder(nn.Module):
    """
    Encoder Block
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck Block
    """

    def __init__(self) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 18 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, 256 * 18 * 18),
            nn.ReLU(),
            nn.Unflatten(1, (256, 18, 18))
        )

    def forward(self, x):
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    """
    Decoder Block
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    """
    Merging three blocks
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    ae = AutoEncoder()
    data = torch.randn((1, 1, 288, 288))
    out = ae(data)
    print(out.shape)

