import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __str__(self) -> str:
        return "SimpleAE"

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    """
    More reliable AutoEncoder
    """

    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.bottleneck = self.conv_block(256, 512)

        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

    def __str__(self) -> str:
        return "UNet"

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
        return block

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Down Sampling (Encoding)
        s1 = self.enc1(x)
        p1 = F.max_pool2d(s1, 2)
        s2 = self.enc2(p1)
        p2 = F.max_pool2d(s2, 2)
        s3 = self.enc3(p2)
        p3 = F.max_pool2d(s3, 2)
        s4 = self.enc4(p3)
        p4 = F.max_pool2d(s4, 2)

        # Base(Bottleneck)
        b1 = self.bottleneck(p4)

        # Up Sampling (Decoding)
        d4 = self.upconv4(b1)
        d4 = torch.cat((d4, s4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, s3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, s2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, s1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.outconv(d1)

        return out
