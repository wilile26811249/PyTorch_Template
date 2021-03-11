import torch
import torch.nn as nn

from .unet_layer import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, bilinear = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.bilinear = bilinear

        self.input_layer = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        feature_factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // feature_factor)
        self.up1 = Up(1024, 512 // feature_factor, self.bilinear)
        self.up2 = Up(512, 256 // feature_factor, self.bilinear)
        self.up3 = Up(256, 128 // feature_factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.output_layer = OutConv(64, self.out_classes)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outputs = self.output_layer(x)
        return outputs