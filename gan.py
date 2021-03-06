import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reversibleconvolution.reversible_convolution import *
from remixer.remixer import *

class Generator(nn.Module):
    """Some Information about Image2Image"""
    def __init__(self, input_channels=3, mid_channels=32, output_channels=3, num_layers=32):
        super(Generator, self).__init__()
        self.channel_conv1 = nn.Conv2d(input_channels, mid_channels, 1, padding=0)
        self.gelu1 = nn.GELU()
        self.rev_conv = ReversibleConvTranspose2d(mid_channels, num_layers=num_layers)
        self.gelu2 = nn.GELU()
        self.channel_conv2 = nn.Conv2d(mid_channels, output_channels, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.channel_conv1(x)
        x = self.gelu1(x)
        x = self.rev_conv(x)
        x = self.gelu2(x)
        x = self.channel_conv2(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, input_channels=3, mid_channels=32, num_layers=16):
        super(Discriminator, self).__init__()
        self.channel_conv1 = nn.Conv2d(input_channels, mid_channels, 1, padding=0)
        self.gelu1 = nn.GELU()
        self.rev_conv = ReversibleConv2d(mid_channels, num_layers=num_layers)
        self.gelu2 = nn.GELU()
        self.remixer = ReMixerImageClassificator(channels=mid_channels, image_size=256, patch_size=16, classes=1, dim=256, num_layers=num_layers)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.channel_conv1(x)
        x = self.gelu1(x)
        x = self.rev_conv(x)
        x = self.gelu2(x)
        x = self.remixer(x)
        x = self.sigmoid(x)
        return x
