import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            out_channels = mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 0),
            nn.BatchNorm2d(num_features = mid_channels),
            nn.ReLU(inplace = True),  # Decrease the memory usage
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 0),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace = True),  # Decrease the memory usage
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        self.maxpool_double_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            DoubleConv(in_channels, out_channels, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear',
                align_corners = True
            )
            self.double_conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size = 2,
                stride = 2
            )
            self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, expansive_in, contracting_prev):
        expansive_up = self.up(expansive_in)
        # Calculate the diff for concatenate (Shape: N x C x H x W)
        diff_x = contracting_prev.size()[3] - expansive_up.size()[3]
        diff_y = contracting_prev.size()[2] - expansive_up.size()[2]
        cropped = contracting_prev[:, :, diff_y : expansive_up.size()[2], diff_x : expansive_up.size()[3]]
        concated = torch.cat([cropped, expansive_up], dim = 1)
        return self.double_conv(concated)
