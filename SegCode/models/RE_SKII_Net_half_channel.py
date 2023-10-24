#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author : hao zhang
# @File   : RE-Net.py
import torch
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out
class ResEncoder_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder_2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels//2)
        self.conv2 = nn.Conv3d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out
class ReDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReDecoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out


class ResDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ResDecoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SKLayer_3D_spatial(nn.Module):
    def __init__(self, in_channel, out_channel, M=2, reduction=4, L=32, G=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
         :param G:  组卷积
        '''
        super(SKLayer_3D_spatial, self).__init__()

        self.M = M
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, 1, padding=1, dilation=1, groups=G, bias=False),
        )
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, padding=1 + i, dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True))
            )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        self.spatial = SpatialAttention()
        d = max(out_channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=out_channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 # nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=out_channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size, channel, _, _, _ = input.shape

        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        # fusion阶段
        U = output[0] + output[1]  # 逐元素相加生成 混合特征U
        s = self.fbap(U)  # B,C,1,1,1
        spatial = self.spatial(U)  # B,1,D,H,W

        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channel, 1, 1, 1), a_b))

        # a_b = list(map(lambda x: x.squeeze(x, dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        V_final = V * spatial + V
        return V_final


class SKLayer_3D(nn.Module):
    def __init__(self, in_channel, out_channel, M=2, reduction=4, L=32, G=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
         :param G:  组卷积
        '''
        super(SKLayer_3D, self).__init__()

        self.M = M
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, padding=1 + i, dilation=1 + i, groups=in_channel, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True))
            )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        # self.spatial = SpatialAttention()
        d = max(out_channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=out_channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 # nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=out_channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size, channel, _, _, _ = input.shape

        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        # fusion阶段
        U = output[0] + output[1]  # 逐元素相加生成 混合特征U
        s = self.fbap(U)  # B,C,1,1,1
        # spatial = self.spatial(U)  # B,1,D,H,W

        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channel, 1, 1, 1), a_b))

        # a_b = list(map(lambda x: x.squeeze(x, dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        # V_final = V * spatial + V
        return V


class SFlayer(nn.Module):
    def __init__(self, out_channel, M=2, reduction=4, L=32, G=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
         :param G:  组卷积
        '''
        super(SFlayer, self).__init__()
        self.out_channel = out_channel
        self.M = M

        # 尺度不变
        # self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        # for i in range(self.M):
        #     # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
        #     self.conv.append(nn.Sequential(
        #         nn.Conv3d(in_channel, out_channel, 3, 1, padding=1 + i, dilation=1 + i, groups=G, bias=False),
        #         nn.BatchNorm3d(out_channel),
        #         nn.ReLU(inplace=True))
        #     )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        # self.spatial = SpatialAttention()
        d = max(out_channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=out_channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 # nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=out_channel, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, x1, x2):
        batch_size, channel, _, _, _ = x1.shape

        # split阶段
        # output = []
        # for i, conv in enumerate(self.conv):
        #     output.append(conv(input))
        #
        # fusion阶段
        output = [x1, x2]
        # output.append()
        U = torch.cat((x1, x2), 1)  # 逐元素相加生成 混合特征U
        s = self.fbap(U)  # B,C,1,1,1
        # spatial = self.spatial(U)  # B,1,D,H,W

        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channel // 2, 1, 1, 1), a_b))

        # a_b = list(map(lambda x: x.squeeze(x, dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = torch.cat((V[0], V[1]), 1)
        # V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        # V_final = V * spatial + V
        return V


class SFConv(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SFConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        # self.convs = nn.ModuleList([])
        # for i in range(M):
        #     self.convs.append(nn.Sequential(
        #         nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
        #         nn.BatchNorm2d(features),
        #         nn.ReLU(inplace=False)
        #     ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # for i, conv in enumerate(self.convs):
        #     fea = conv(x).unsqueeze_(dim=1)
        #     if i == 0:
        #         feas = fea
        #     else:
        #         feas = torch.cat([feas, fea], dim=1)
        feas = torch.cat((x1.unsqueeze_(dim=1), x2.unsqueeze_(dim=1)), dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKII_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKII_Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = SKLayer_3D(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out


class SF_Decoder(nn.Module):
    def __init__(self, out_channels):
        super(SF_Decoder, self).__init__()
        self.conv1 = SFConv(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        # self.conv2 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(out_channels // 2)
        self.relu = nn.ReLU(inplace=False)
        self.ResDecoder = ResDecoder(out_channels)
        # self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x1, x2)))
        out = self.ResDecoder(out)

        # out = self.relu(self.bn2(self.conv2(out)))
        # out += residual
        # out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class RE_Net_SK_docoder(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(RE_Net_SK_docoder, self).__init__()
        self.encoder1 = ResEncoder(channels, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.bridge = ResEncoder(256, 512)

        self.conv1_1 = nn.Conv3d(512, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(128, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = SF_Decoder(256)
        self.decoder2 = SF_Decoder(128)
        self.decoder1 = SF_Decoder(64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        # up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3, x)

        up2 = self.up2(dec3)
        # up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2, x2)

        up1 = self.up1(dec2)
        # up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1, x3)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final


class RE_Net_SK_docoder_half_channel(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(RE_Net_SK_docoder_half_channel, self).__init__()
        self.encoder1 = ResEncoder(channels, 32)
        self.encoder2 = ResEncoder(32, 64)
        self.encoder3 = ResEncoder(64, 128)
        self.bridge = ResEncoder(128, 256)

        self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = SF_Decoder(128)
        self.decoder2 = SF_Decoder(64)
        self.decoder1 = SF_Decoder(32)
        self.down = downsample()
        self.up3 = deconv(256, 128)
        self.up2 = deconv(128, 64)
        self.up1 = deconv(64, 32)
        self.final = nn.Conv3d(32, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        # up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3, x)

        up2 = self.up2(dec3)
        # up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2, x2)

        up1 = self.up1(dec2)
        # up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1, x3)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final

class RE_Net(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(RE_Net, self).__init__()
        self.encoder1 = ResEncoder_2(channels, 64)
        self.encoder2 = ResEncoder_2(64, 128)
        self.encoder3 = ResEncoder_2(128, 256)
        self.bridge = ResEncoder_2(256, 512)

        self.conv1_1 = nn.Conv3d(512, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(128, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = ReDecoder(512, 256)
        self.decoder2 = ReDecoder(256, 128)
        self.decoder1 = ReDecoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class RE_Net_SK(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(RE_Net_SK, self).__init__()
        self.encoder1 = ResEncoder_2(channels, 64)
        self.encoder2 = ResEncoder_2(64, 128)
        self.encoder3 = ResEncoder_2(128, 256)
        self.bridge = ResEncoder_2(256, 512)

        self.conv1_1 = nn.Conv3d(512, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(128, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = ReDecoder(512, 256)
        self.decoder2 = ReDecoder(256, 128)
        self.decoder1 = ReDecoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class RE_Net_half_channel(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(RE_Net_half_channel, self).__init__()
        self.encoder1 = ResEncoder(channels, 32)
        self.encoder2 = ResEncoder(32, 64)
        self.encoder3 = ResEncoder(64, 128)
        self.bridge = ResEncoder(128, 256)

        self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = ReDecoder(256, 128)
        self.decoder2 = ReDecoder(128, 64)
        self.decoder1 = ReDecoder(64, 32)
        self.down = downsample()
        self.up3 = deconv(256, 128)
        self.up2 = deconv(128, 64)
        self.up1 = deconv(64, 32)
        self.final = nn.Conv3d(32, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        con3_3 = self.conv3_3(enc2)
        convTrans3 = self.convTrans3(con3_3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        con2_2 = self.conv2_2(enc3)
        convTrans2 = self.convTrans2(con2_2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        bridge = self.bridge(down3)

        conv1_1 = self.conv1_1(bridge)
        convTrans1 = self.convTrans1(conv1_1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

        up3 = torch.cat((up3, x), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class ResUet_half_channel(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(ResUet_half_channel, self).__init__()
        self.encoder1 = ResEncoder(channels, 32)
        self.encoder2 = ResEncoder(32, 64)
        self.encoder3 = ResEncoder(64, 128)
        self.bridge = ResEncoder(128, 256)

        # self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        # self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        # self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)
        #
        # self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        # self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        # self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = ReDecoder(256, 128)
        self.decoder2 = ReDecoder(128, 64)
        self.decoder1 = ReDecoder(64, 32)
        self.down = downsample()
        self.up3 = deconv(256, 128)
        self.up2 = deconv(128, 64)
        self.up1 = deconv(64, 32)
        self.final = nn.Conv3d(32, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        # con3_3 = self.conv3_3(enc2)
        # convTrans3 = self.convTrans3(con3_3)
        # x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        # x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        # x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        # con2_2 = self.conv2_2(enc3)
        # convTrans2 = self.convTrans2(con2_2)
        # x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        # x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        # x2 = x2 + enc2

        bridge = self.bridge(down3)

        # conv1_1 = self.conv1_1(bridge)
        # convTrans1 = self.convTrans1(conv1_1)
        #
        # x = -1 * (torch.sigmoid(convTrans1)) + 1
        # x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        # x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)

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
################ ResUnet############


class ResUnet(nn.Module):
    def __init__(self, classes, channels):
        # def __init__(self):

        super(ResUnet, self).__init__()
        self.encoder1 = ResEncoder_2(channels, 64)
        self.encoder2 = ResEncoder_2(64, 128)
        self.encoder3 = ResEncoder_2(128, 256)
        self.bridge = ResEncoder_2(256, 512)

        # self.conv1_1 = nn.Conv3d(256, 1, kernel_size=1)
        # self.conv2_2 = nn.Conv3d(128, 1, kernel_size=1)
        # self.conv3_3 = nn.Conv3d(64, 1, kernel_size=1)
        #
        # self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        # self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        # self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = ReDecoder(512, 256)
        self.decoder2 = ReDecoder(256, 128)
        self.decoder1 = ReDecoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, classes, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        # con3_3 = self.conv3_3(enc2)
        # convTrans3 = self.convTrans3(con3_3)
        # x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        # x3 = x3.expand(-1, 32, -1, -1, -1).mul(enc1)
        # x3 = x3 + enc1

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        # con2_2 = self.conv2_2(enc3)
        # convTrans2 = self.convTrans2(con2_2)
        # x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        # x2 = x2.expand(-1, 64, -1, -1, -1).mul(enc2)
        # x2 = x2 + enc2

        bridge = self.bridge(down3)

        # conv1_1 = self.conv1_1(bridge)
        # convTrans1 = self.convTrans1(conv1_1)
        #
        # x = -1 * (torch.sigmoid(convTrans1)) + 1
        # x = x.expand(-1, 128, -1, -1, -1).mul(enc3)
        # x = x + enc3

        up3 = self.up3(bridge)
        # up3 = SKII_Decoder(up3,)
        print(up3.shape,enc3.shape)
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