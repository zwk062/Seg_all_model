"""Split-Attention"""
from resnest.torch.resnet import DropBlock2D

"""
Reference:

- Zhang, Hang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun et al. "Resnest: Split-attention networks." arXiv preprint arXiv:2004.08955 (2020)
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair, _triple

# from .resnet import RFConv3d
class DropBlock3D(nn.Module):
    def __init__(self, keep_prob, block_size):
        super(DropBlock3D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.keep_prob == 1.0:
            return x

        # Get the spatial dimensions of the input
        _, _, d, h, w = x.size()

        # Calculate gamma
        gamma = self.calculate_gamma(x)

        # Create a binary mask
        mask = (torch.rand(x.size()) < gamma).float()

        # Calculate block mask
        block_mask = self.calculate_block_mask(mask)

        # Apply block mask to the input
        out = x * block_mask[:, :, None, :, :]

        # Adjust the output to account for the dropped values
        out = out * block_mask.numel() / block_mask.sum()

        return out
    def calculate_gamma(self, x):
        return self.keep_prob / (self.block_size ** 3) * (x.size(2) * x.size(3) * x.size(4)) / \
               ((x.size(2) - self.block_size + 1) * (x.size(3) - self.block_size + 1) * (x.size(4) - self.block_size + 1))

    def calculate_block_mask(self, mask):
        block_mask = -torch.nn.functional.max_pool3d(-mask,
                                                      kernel_size=(self.block_size, self.block_size, self.block_size),
                                                      stride=(1, 1, 1),
                                                      padding=self.block_size // 2)
        return 1 - block_mask

class RFConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth, stride=1, padding=0):
        """
        Recursive Feature Convolution for 3D data
        Args:
            in_channels (int): 输入的通道数
            out_channels (int): 输出的通道数
            kernel_size (int): 卷积核的尺寸（深度、高度、宽度都是 kernel_size）
            depth (int): 递归深度，即递归应用卷积核的次数
            stride (int): 卷积的步幅 (默认为1)
            padding (int): 卷积的填充 (默认为0)
        """
        super(RFConv3d, self).__init__()
        self.depth = depth
        self.conv_layers = nn.ModuleList()

        for _ in range(depth):
            self.conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels  # 更新输入通道数，以便下一层使用

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

__all__ = ['SplAtConv2d']

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation= 1, groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=nn.BatchNorm3d,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _triple(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0 or padding[2] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            # from rfconv import RFConv2d

            self.conv = RFConv3d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv3d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
            self.bn2 = norm_layer(channels)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv3d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock3D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)
        self.conv2 = Conv3d(channels, channels, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.dropout = nn.Dropout3d(0.2)
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        # print("rchannel dimensions:", rchannel)
        # print(rchannel // self.radix)
        x1, x2 = torch.split(x, rchannel//self.radix, dim=1)
        # print("x1 dimensions:", x1.size())
        # print("x2 dimensions:", x2.size())
        x2 = x2 + x1
        
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        # print("x2 dimensions:", x2.size())
        splited = (x1, x2)
        # print("splited dimensions:", splited[0].size())
        # print("splited dimensions:", splited[1].size())
        gap = sum(splited) 
       
        gap = F.adaptive_avg_pool3d(gap, (1,1,1))
        gap = self.fc1(gap)
        # print("gap dimensions:", gap.size())
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        # print("gap dimensions:", gap.size())
        atten = self.fc2(gap)
        # print("atten dimensions:", atten.size())
        # atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        # print("atten dimensions:", atten.size())
        attens = torch.split(atten, rchannel//self.radix, dim=1)
        # print("attens dimensions:", attens[0].size())

        out = sum([att*split for (att, split) in zip(attens, splited)])
        return out.contiguous()

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
