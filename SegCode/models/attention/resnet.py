"""ResNet variants"""
import math
import torch
import torch.nn as nn

# from pytorch_dcsaunet import splat
from .splat import SplAtConv2d
#
__all__ = ['ResNet', 'Bottleneck']


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


# class DropBlock2D(object):
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool3d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool3d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=nn.BatchNorm3d, dropblock_prob=0.0, last_gamma=False,number=1,custom=0):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        if custom != 0:
            inplanes = custom
        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock3D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock3D(dropblock_prob, 3)
            self.dropblock3 = DropBlock3D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            # from rfconv import RFConv2d
            self.conv2 = RFConv3d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv3d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv3d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        #self.dp = nn.Dropout(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.dp(out)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)
            
        
        
        out = self.conv2(out)
        #out = self.dp(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
            
        

        out = self.conv3(out)
       # out = self.dp(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=2,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm3d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width*2
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        
        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            import RFConv3d
            conv_layer = RFConv3d
        else:
            conv_layer = nn.Conv3d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        ConvFea = [32,64,128,256,512]
        self.layer1 = self._make_layer(block, ConvFea[0], layers[0], norm_layer=norm_layer, is_first=False)
        
        self.layer2 = self._make_layer(block, ConvFea[1], layers[1], stride=1, norm_layer=norm_layer)
        
            

        self.layer3 = self._make_layer(block, ConvFea[2], layers[2], stride=1,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        
        self.layer4 = self._make_layer(block, ConvFea[2], layers[3], stride=1,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
           
            
        self.layer5 = self._make_layer(block, ConvFea[1], layers[0], stride=1,
                                            dilation=1, norm_layer=norm_layer,
                                            dropblock_prob=dropblock_prob,inchannel=1024)
        self.layer6 = self._make_layer(block, ConvFea[0], layers[1], stride=1,
                                            dilation=1, norm_layer=norm_layer,
                                            dropblock_prob=dropblock_prob,inchannel=512)
        self.layer7 = self._make_layer(block, ConvFea[0] // 2, layers[2], stride=1, norm_layer=norm_layer,inchannel=256)
        self.layer8 = self._make_layer(block, ConvFea[0] // 2, layers[3], norm_layer=norm_layer, is_first=False,inchannel=128) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True,inchannel=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or inchannel != 0:
            if inchannel != 0:
                self.inplanes = inchannel
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool3d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool3d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv3d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv3d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,custom=inchannel))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)
