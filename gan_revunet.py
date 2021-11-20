import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reversibleconvolution.reversible_convolution import *
from remixer.remixer import *

class DownConv(nn.Module):
    """Some Information about DownConv"""
    def __init__(self, input_channels, output_channels, num_layers=1):
        super(DownConv, self).__init__()
        self.reconv = ReversibleConv2d(input_channels, groups=1, num_layers=num_layers)
        self.pool = nn.MaxPool2d(2)
        self.channel_conv = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        x = self.reconv(x)
        x = self.pool(x)
        x = self.channel_conv(x)
        return x

class UpConv(nn.Module):
    """Some Information about UpConv"""
    def __init__(self, input_channels, output_channels, num_layers=1):
        super(UpConv, self).__init__()
        self.channel_conv = nn.Conv2d(input_channels, output_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.reconv = ReversibleConv2d(output_channels, groups=1, num_layers=num_layers)
    def forward(self, x):
        x = self.channel_conv(x)
        x = self.upsample(x)
        x = self.reconv(x)
        return x


class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, num_layers=3):
        super(Generator, self).__init__()
        self.ch_conv_in = nn.Conv2d(3, 8, 1)
        self.down1 = DownConv(8, 16, num_layers=num_layers) 
        self.down2 = DownConv(16, 32, num_layers=num_layers) 
        self.down3 = DownConv(32, 64, num_layers=num_layers) 
        self.down4 = DownConv(64, 64, num_layers=num_layers) 
        self.down5 = DownConv(64, 64, num_layers=num_layers) 
        self.down6 = DownConv(64, 96, num_layers=num_layers) 

        self.up1 = UpConv(96, 64, num_layers=num_layers) 
        self.up2 = UpConv(64, 64, num_layers=num_layers) 
        self.up3 = UpConv(64, 64, num_layers=num_layers) 
        self.up4 = UpConv(64, 32, num_layers=num_layers) 
        self.up5 = UpConv(32, 16, num_layers=num_layers) 
        self.up6 = UpConv(16, 8, num_layers=num_layers) 
        self.ch_conv_out = nn.Conv2d(8, 3, 1)
    def forward(self, x):
        x1 = self.ch_conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.up1(x7)
        x9 = self.up2(x8 + x6)
        x10 = self.up3(x9 + x5)
        x11 = self.up4(x10 + x4)
        x12 = self.up5(x11 + x3)
        x13 = self.up6(x12 + x2)
        x14 = self.ch_conv_out(x13 + x1)
        return x14


class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, num_layers=3):
        super(Discriminator, self).__init__()
        self.ch_conv_in = nn.Conv2d(3, 4, 1)
        self.down1 = DownConv(4, 8, num_layers=num_layers) 
        self.down2 = DownConv(8, 16, num_layers=num_layers) 
        self.down3 = DownConv(16, 32, num_layers=num_layers) 
        self.down4 = DownConv(32, 64, num_layers=num_layers) 
        self.down5 = DownConv(64, 64, num_layers=num_layers) 
        self.down6 = DownConv(64, 64, num_layers=num_layers) 
        self.down7 = DownConv(64, 128, num_layers=num_layers) 
        self.down8 = DownConv(128, 128, num_layers=num_layers) 
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.ch_conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x = self.down8(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x