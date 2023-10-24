# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv3d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair


__all__ = ['SAUNet']
def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv3d(Module):
    """Split-Attention Conv3d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=(1, 1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv3d, self).__init__()
        #padding = _pair(padding)         ###
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0 or padding[2] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        # if self.rectify:
        # # # from rfconv import RFConv2d
        # # self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
        #     #                     groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        # else:
        self.conv = Conv3d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv3d(inter_channels, channels*radix, 1, groups=self.cardinality)
        # if dropblock_prob > 0.0:
        #     self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        # if self.use_bn:
        #     gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()




class Splitblock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Splitblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            SplAtConv3d(ch_out, ch_out, kernel_size=3, padding=1, groups=2, norm_layer=nn.BatchNorm3d)
        )
        self.downsample = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(ch_out),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)




class SAUNet(nn.Module):
    def __init__(self):

        super(SAUNet, self).__init__()
        self.encoder1 = Splitblock(1, 64)
        self.encoder2 = Splitblock(64, 128)
        self.encoder3 = Splitblock(128, 256)
        self.bridge = Splitblock(256, 512)
        self.decoder3 = Splitblock(512, 256)
        self.decoder2 = Splitblock(256, 128)
        self.decoder1 = Splitblock(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        bridge = self.bridge(down3)

        up3 = self.up3(bridge)
        up3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final

