import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# UNet Model
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()

        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.d5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv_up4 = DoubleConv(1024, 512)
        self.conv_up3 = DoubleConv(512, 256)
        self.conv_up2 = DoubleConv(256, 128)
        self.conv_up1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        x1 = self.d1(x)  # 64
        x2 = self.d2(self.pool(x1))  # 128
        x3 = self.d3(self.pool(x2))  # 256
        x4 = self.d4(self.pool(x3))  # 512
        x5 = self.d5(self.pool(x4))  # 1024

        # Decoder
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.conv_up4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv_up3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv_up1(d1)

        out = self.final(d1)
        return out
