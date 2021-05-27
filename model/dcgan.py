import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim = 64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 5,
                        stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, dim * 16 * 4 * 4, bias = False),
            nn.BatchNorm1d(dim * 16 * 4 * 4),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            dconv_bn_relu(dim * 16, dim * 8),   # (b, 512, 8, 8)
            dconv_bn_relu(dim * 8, dim * 4),   # (b, 256, 16, 16)
            dconv_bn_relu(dim * 4, dim * 2),   # (b, 128, 32, 32)
            nn.ConvTranspose2d(dim * 2, 3, kernel_size = 5,
                    stride = 2, padding = 2, output_padding = 1), # (b, 3, 64, 64)
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1, 4, 4) # (b, 1024, 4, 4)
        x = self.layer2(x)
        return x


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N,)
    """
    def __init__(self, in_dim, dim = 64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1)
        return x
