import numpy as np
import torch.nn as nn


class AttDecoder(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(AttDecoder, self).__init__()
        self.aconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,out_channels,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        out1 = np.dot(self.aconv(x),x)
        out1 += x
        return out1