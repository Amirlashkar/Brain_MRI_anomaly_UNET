import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from constants import *


class ImageDataset(Dataset):
    """
    Making data accessible
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


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


class UNet(nn.Module):
    def conv_block(self, input:torch.Tensor, in_channels:int, out_channels:int):
        x = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)(input)
        x = nn.BatchNorm2d(out_channels)(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(.25)(x)
        x = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)(x)
        x = nn.BatchNorm2d(out_channels)(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(.25)(x)

        return x

    def upconv(self, input:torch.Tensor, in_channels:int, out_channels:int):
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)(input)
        return up

    def encoder_block(self, input:torch.Tensor, in_channels:int, out_channels:int):
        s = self.conv_block(input, in_channels, out_channels)
        p = F.max_pool2d(s, 2)

        return s, p

    def decoder_block(self, input:torch.Tensor, skip_features:torch.Tensor, in_channels:int, out_channels:int):
        d = self.upconv(input, in_channels, out_channels)
        d = torch.cat((d, skip_features), dim=1)
        d = self.conv_block(d, in_channels, out_channels)

        return d

    def forward(self, x):
        # Down Sampling (Encoding)
        s1, p1 = self.encoder_block(x, 1, 64)
        s2, p2 = self.encoder_block(p1, 64, 128)
        s3, p3 = self.encoder_block(p2, 128, 256)
        s4, p4 = self.encoder_block(p3, 256, 512)

        # Base(Bottleneck)
        b1 = self.conv_block(p4, 512, 1024)

        # Up Sampling (Decoding)
        d4 = self.decoder_block(b1, s4, 1024, 512)
        d3 = self.decoder_block(d4, s3, 512, 256)
        d2 = self.decoder_block(d3, s2, 256, 128)
        d1 = self.decoder_block(d2, s1, 128, 64)

        # Output
        out = nn.Conv2d(64, 1, kernel_size=1)(d1)

        return out

if __name__ == "__main__":
    ae = AutoEncoder()
    data = torch.randn((1, 1, 288, 288))
    out = ae(data)
    print(out.shape)

