import random

import torch.nn as nn
import torch
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.nn.functional as F
from .attention.attention import AttDecoder

# 最大池化
def downsample():
    # return nn.MaxPool3d(kernel_size=2, stride=2)
    return nn.AvgPool3d(kernel_size=2, stride=2)
#卷积
def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2))

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

class Reduction_Encoder(nn.Module):
    def __init__(self, input, c1,  c2, c3,**kwargs):
        super(Reduction_Encoder, self).__init__(**kwargs)
        # 线路为1×1×1、5×5×5（步长为2）的卷积链
        # 5*5*5的卷积padding应该是2
        # [64-5+4]/2+1 =32
        self.p1 = nn.Sequential(
            nn.Conv3d(input, c1[0], kernel_size=(1, 1, 1), padding=(2, 2, 2)),
            nn.BatchNorm3d(c1[0]),
            nn.ReLU(),
            nn.Conv3d(c1[0], c1[1], kernel_size=(5, 5, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(c1[1]),
            nn.ReLU()
        )
        # 路径为MaxPooling,1×1×1的路径
        # self.p2 = nn.Sequential(
        #     nn.Conv3d(input, c2, kernel_size=(1, 1, 1)),
        #     nn.ReLU()
        # )
        # 路径为1×1×1（步长为2、1×1×5、1×5×1、5×1×1
        self.p2 = nn.Sequential(
            nn.Conv3d(input, c2[0], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(2, 2, 2)),
            nn.BatchNorm3d(c2[0]),
            nn.ReLU(),
            nn.Conv3d(c2[0], c2[1], kernel_size=(1, 1, 5)),
            nn.BatchNorm3d(c2[1]),
            nn.ReLU(),
            nn.Conv3d(c2[1], c2[2], kernel_size=(1, 5, 1)),
            nn.BatchNorm3d(c2[2]),
            nn.ReLU(),
            nn.Conv3d(c2[2], c2[3], kernel_size=(5, 1, 1)),
            nn.BatchNorm3d(c2[3]),
            nn.ReLU()

        )
        # self.p3 = nn.Sequential(
        #     nn.Conv3d(input, c3[0], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        #     nn.BatchNorm3d(c3[0]),
        #     nn.ReLU(),
        #     nn.Conv3d(c3[0], c3[1], kernel_size=(5, 5, 1), padding=(2, 2, 0)),
        #     nn.BatchNorm3d(c3[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(c3[1], c3[2], kernel_size=(5, 1, 5), padding=(2, 0, 2)),
        #     nn.BatchNorm3d(c3[2]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(c3[2], c3[3], kernel_size=(1, 5, 5), padding=(0, 2, 2)),
        #     nn.BatchNorm3d(c3[3]),
        #     nn.ReLU(inplace=True),
        # )

        self.p3 = nn.Sequential(
            nn.Conv3d(input, c3, kernel_size=(1, 1, 1)),
            # nn.ReLU()
        )

        self.relu =nn.ReLU()


    def forward(self, x):
        p1 = self.p1(x)
        # p2 = self.p2(x)
        p2 = self.p2(x)

        res = torch.cat((p1, p2), dim=1)
        p3 = self.p3(x)
        res = p3 + res
        res = self.relu(res)
        return res


# class Reduction_Encoder(nn.Module):
#     def __init__(self, input, c1, c2, c3,c4,c5, **kwargs):
#         super(Reduction_Encoder, self).__init__(**kwargs)
#         # 线路为1×1×1、5×5×5（步长为2）的卷积链
#         # 5*5*5的卷积padding应该是2
#         # [64-5+4]/2+1 =32
#         self.p1 = nn.Sequential(
#             nn.Conv3d(input, c1[0], kernel_size=(1, 1, 1), padding=(2, 2, 2)),
#             nn.BatchNorm3d(c1[0]),
#             nn.Sigmoid(),
#             nn.Conv3d(c1[0], c1[1], kernel_size=(5, 5, 5), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c1[1]),
#             nn.ReLU()
#         )
#         # 路径为MaxPooling,1×1×1的路径
#         # self.p2 = nn.Sequential(
#         #     nn.Conv3d(input, c2, kernel_size=(1, 1, 1)),
#         #     nn.ReLU()
#         # )
#         # 路径为1×1×1（步长为2、1×1×5、1×5×1、5×1×1
#         self.p2 = nn.Sequential(
#             nn.Conv3d(input, c2[0], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(2, 2, 2)),
#             nn.BatchNorm3d(c2[0]),
#             nn.ReLU(),
#             nn.Conv3d(c2[0], c2[1], kernel_size=(1, 1, 5)),
#             nn.BatchNorm3d(c2[1]),
#             nn.ReLU(),
#             nn.Conv3d(c2[1], c2[2], kernel_size=(1, 5, 1)),
#             nn.BatchNorm3d(c2[2]),
#             nn.ReLU(),
#             nn.Conv3d(c2[2], c2[3], kernel_size=(5, 1, 1)),
#             nn.BatchNorm3d(c2[3]),
#             nn.ReLU()
#
#         )
#         self.p3 = nn.Sequential(
#             nn.Conv3d(input, c3[0], kernel_size=(1, 1, 1),padding=(1, 1, 1)),
#             nn.BatchNorm3d(c3[0]),
#             nn.Sigmoid(),
#             nn.Conv3d(c3[0], c3[1], kernel_size=(3, 3, 3), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c3[1]),
#             nn.ReLU()
#         )
#         self.p4 = nn.Sequential(
#             nn.Conv3d(input, c4[0], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(1, 1, 1)),
#             nn.BatchNorm3d(c4[0]),
#             nn.ReLU(),
#             nn.Conv3d(c4[0], c4[1], kernel_size=(1, 1, 3)),
#             nn.BatchNorm3d(c4[1]),
#             nn.ReLU(),
#             nn.Conv3d(c4[1], c4[2], kernel_size=(1, 3, 1)),
#             nn.BatchNorm3d(c4[2]),
#             nn.ReLU(),
#             nn.Conv3d(c4[2], c4[3], kernel_size=(3, 1, 1)),
#             nn.BatchNorm3d(c4[3]),
#             nn.ReLU()
#
#
#         )
#
#         self.p5 = nn.Sequential(
#             nn.Conv3d(input, c5, kernel_size=(1, 1, 1)),
#             # nn.ReLU()
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         p1 = self.p1(x)
#         # print(p1.shape)
#         # p2 = self.p2(x)
#         p2 = self.p2(x)
#         # print(p2.shape)
#         p3 = self.p3(x)
#         # print(p3.shape)
#         p4 =self.p4(x)
#         # print(p4.shape)
#         p5 =self.p5(x)
#         # print(p5.shape)
#         res = torch.cat((p1,p2,p3,p4), dim=1)
#         res = p5 + res
#         res = self.relu(res)
#         return res

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
        self.res = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            # nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        res = self.res(x)
        out = res + out
        out = self.relu(out)
        return out

# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         # self.in1 = nn.InstanceNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         # self.in2 = nn.InstanceNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=False)
#         self.res = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         residual = self.res(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out += residual
#         out = self.relu(self.bn2(out))
#         return out

# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.conv = nn.Sequential(
#             # nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
#             # nn.BatchNorm3d(out_channels),
#             # nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 1), padding=(2, 2, 0)),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=(5, 1, 5), padding=(1, 0, 1)),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.res = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
#             # nn.ReLU()
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         res =self.res(x)
#         out = res + out
#         out = self.relu(out)
#         return out


# class Decoder(nn.Module):
#     def __init__(self, input, c1,  c2, c3, c4, **kwargs):
#         super(Decoder, self).__init__(**kwargs)
#         # 线路为1×1×1、5×5×5（步长为2）的卷积链
#         # 5*5*5的卷积padding应该是2
#         # [64-5+4]/2+1 =32
#         self.p1 = nn.Sequential(
#             nn.Conv3d(input, c1[0], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c1[0]),
#             nn.ReLU(),
#             nn.Conv3d(c1[0], c1[1], kernel_size=(5, 5, 1),  padding=(2, 2, 0)),
#             nn.BatchNorm3d(c1[1]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(c1[1], c1[2], kernel_size=(5, 5, 5),padding=(2, 2, 2)),
#             nn.BatchNorm3d(c1[2]),
#             nn.ReLU(inplace=True),
#         )
#
#         # 路径为1×1×1（步长为2、1×1×5、1×5×1、5×1×1
#         self.p2 = nn.Sequential(
#             nn.Conv3d(input, c2[0], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c2[0]),
#             nn.ReLU(),
#             nn.Conv3d(c2[0], c2[1], kernel_size=(5, 1, 5), padding=(2, 0, 2)),
#             nn.BatchNorm3d(c2[1]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(c2[1], c2[2], kernel_size=(5, 5, 5), padding=(2, 2, 2)),
#             nn.BatchNorm3d(c2[2]),
#             nn.ReLU(inplace=True),
#         )
#         self.p3 = nn.Sequential(
#             nn.Conv3d(input, c3[0], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c3[0]),
#             nn.ReLU(),
#             nn.Conv3d(c3[0], c3[1], kernel_size=(1, 5, 5), padding=(0, 2, 2)),
#             nn.BatchNorm3d(c3[1]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(c3[1], c3[2], kernel_size=(5, 5, 5), padding=(2, 2, 2)),
#             nn.BatchNorm3d(c3[2]),
#             nn.ReLU(inplace=True),
#         )
#
#         self.p4 = nn.Sequential(
#             nn.Conv3d(input, c4, kernel_size=(1, 1, 1)),
#             # nn.ReLU()
#         )
#         self.relu =nn.ReLU()
#
#
#     def forward(self, x):
#         p1 = self.p1(x)
#         # print(p1.shape)
#         p2 = self.p2(x)
#         # print()
#         p3 =self.p3(x)
#         res = torch.cat((p1, p2, p3), dim=1)
#         p4 = self.p4(x)
#         res = p4 + res
#         res = self.relu(res)
#         return res

# class Reduction_Dncoder(nn.Module):
#     def __init__(self, input, c1,  c2, c3,**kwargs):
#         super(Reduction_Dncoder, self).__init__(**kwargs)
#         # 线路为1×1×1、5×5×5（步长为2）的卷积链
#         # 5*5*5的卷积padding应该是2
#         # [64-5+4]/2+1 =32
#         self.p1 = nn.Sequential(
#             nn.Conv3d(input, c1[0], kernel_size=(1, 1, 1), padding=(2, 2, 2)),
#             nn.BatchNorm3d(c1[0]),
#             nn.ReLU(),
#             nn.Conv3d(c1[0], c1[1], kernel_size=(3, 3, 3), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c1[1]),
#             nn.ReLU(),
#             nn.Conv3d(c1[1], c1[2], kernel_size=(3, 3, 3), stride=(1, 1, 1)),
#             nn.BatchNorm3d(c1[2]),
#             nn.ReLU()
#         )
#         # 路径为MaxPooling,1×1×1的路径
#         # self.p2 = nn.Sequential(
#         #     nn.Conv3d(input, c2, kernel_size=(1, 1, 1)),
#         #     nn.ReLU()
#         # )
#         # 路径为1×1×1（步长为2、1×1×5、1×5×1、5×1×1
#         self.p2 = nn.Sequential(
#             nn.Conv3d(input, c2[0], kernel_size=(1, 1, 1), stride=(1, 1, 1),padding=(2,2,2)),
#             nn.BatchNorm3d(c2[0]),
#             nn.ReLU(),
#             nn.Conv3d(c2[0], c2[1], kernel_size=(1, 1, 5)),
#             nn.BatchNorm3d(c2[1]),
#             nn.ReLU(),
#             nn.Conv3d(c2[1], c2[2], kernel_size=(1, 5, 1)),
#             nn.BatchNorm3d(c2[2]),
#             nn.ReLU(),
#             nn.Conv3d(c2[2], c2[3], kernel_size=(5, 1, 1)),
#             nn.BatchNorm3d(c2[3]),
#             nn.ReLU()
#
#         )
#         # self.p3 = nn.Sequential(
#         #     nn.Conv3d(input, c3[0], kernel_size=(1, 1, 1), stride=(1, 1, 1)),
#         #     nn.BatchNorm3d(c3[0]),
#         #     nn.ReLU(),
#         #     nn.Conv3d(c3[0], c3[1], kernel_size=(5, 5, 1), padding=(2, 2, 0)),
#         #     nn.BatchNorm3d(c3[1]),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(c3[1], c3[2], kernel_size=(5, 1, 5), padding=(2, 0, 2)),
#         #     nn.BatchNorm3d(c3[2]),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(c3[2], c3[3], kernel_size=(1, 5, 5), padding=(0, 2, 2)),
#         #     nn.BatchNorm3d(c3[3]),
#         #     nn.ReLU(inplace=True),
#         # )
#
#         self.p3 = nn.Sequential(
#             nn.Conv3d(input, c3, kernel_size=(1, 1, 1)),
#             # nn.ReLU()
#         )
#
#         self.relu =nn.ReLU()
#
#
#     def forward(self, x):
#         p1 = self.p1(x)
#         # p2 = self.p2(x)
#         p2 = self.p2(x)
#
#         res = torch.cat((p1, p2), dim=1)
#         # print(res.shape)
#         p3 = self.p3(x)
#         # print(p3.shape)
#         res = p3 + res
#
#         res = self.relu(res)
#         return res

class Re_mui_net(nn.Module):
        def __init__(self):
            super(Re_mui_net, self).__init__()

            #五
            self.encoder1 = Reduction_Encoder(1, (16, 32), (16, 32, 32, 32), 64)
            self.encoder2 = Reduction_Encoder(64, (32, 64), (32, 64, 64, 64), 128)
            self.encoder3 = Reduction_Encoder(128, (64, 128),  (64, 128, 128, 128), 256)
            self.bridge = Reduction_Encoder(256, (128, 256),  (128, 256, 256, 256), 512)

            #三和五双重组合模块
            # self.encoder1 = Reduction_Encoder(1, (8, 16), (8, 16, 16, 16), (8, 16), (8, 16, 16, 16), 64)
            # self.encoder2 = Reduction_Encoder(64, (16, 32), (16, 32, 32, 32), (16, 32), (16, 32, 32, 32), 128)
            # self.encoder3 = Reduction_Encoder(128, (32, 64), (32, 64, 64, 64), (32, 64), (32, 64, 64, 64), 256)
            # self.bridge = Reduction_Encoder(256, (64, 128), (64, 128, 128, 128), (64, 128), (64, 128, 128, 128), 512)

            # self.encoder1 = Encoder(1,64)
            # self.encoder2 = Encoder(64, 128)
            # self.encoder3 = Encoder(128, 256)
            # self.bridge = Encoder(256, 512)

            self.weight1 = nn.Parameter(torch.randn(96, 96, 96))
            # self.weight2 = nn.Parameter(torch.randn(96, 96, 96))
            self.weight3 = nn.Parameter(torch.randn(48, 48, 48))
            # self.weight4 = nn.Parameter(torch.randn(48, 48, 48))
            self.weight5 = nn.Parameter(torch.randn(24, 24, 24))
            # self.weight6 = nn.Parameter(torch.randn(24, 24, 24))


            self.conv1_1 = nn.Conv3d(512, 1, kernel_size=(1, 1, 1))
            self.conv2_2 = nn.Conv3d(256, 1, kernel_size=(1, 1, 1))
            self.conv3_3 = nn.Conv3d(128, 1, kernel_size=(1, 1, 1))

            self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            # self.decoder3 = Decoder(512, (256, 128, 128), (128, 64, 64), (128, 64, 64), 256)
            # self.decoder2 = Decoder(256, (128, 64, 64), (64, 32, 32), (64, 32, 32), 128)
            # self.decoder1 = Decoder(128, (64, 32, 32), (32, 16, 16), (32, 16, 16), 64)
            self.decoder3 = Decoder(512,  256)
            self.decoder2 = Decoder(256,  128)
            self.decoder1 = Decoder(128,  64)
            self.down = downsample()


            self.up3 = deconv(512, 256)



            self.up2 = deconv(256, 128)

            # initialize_weights(self)

            self.up1 = deconv(128, 64)



            self.final = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))


        def forward(self, x):
            enc1 = self.encoder1(x)
            # print(enc1.shape)
            # print(enc1.shape)
            down1 = self.down(enc1)
            # print(down1.shape)
            enc2 = self.encoder2(down1)
            # print(enc2.shape)
            down2 = self.down(enc2)
            # print(down2.shape)
            con3_3 = self.conv3_3(enc2)
            # print(con3_3.shape)
            convTrans3 = self.convTrans3(con3_3)
            # print(convTrans3.shape)
            x3 = -1 * (torch.sigmoid(convTrans3)) + 1
            # print(x3.shape)
            # print(enc1.shape)
            # print(x3.shape)
            # x3 = 0.5*x3
            # enc1 =0.5*enc1

            #颜色反转和反转之前进行相乘来提取血管边缘特征
            # x31 = x3.mul(torch.sigmoid(convTrans3))
            #加入权重变量
            x3 = self.weight1.mul(x3)
            # print(x3.shape)
            # enc1 = self.weight2.mul(enc1)
            # x3 = torch.relu(convTrans3)
            # print(x3.shape)
            #加了个sigmoid试一试
            x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
            # print(x3.shape)
            x3 = x3 + enc1
            # x3 = 1 - x3
            # print(x3.shape)
            enc3 = self.encoder3(down2)
            # print(enc3.shape)
            down3 = self.down(enc3)

            con2_2 = self.conv2_2(enc3)
            convTrans2 = self.convTrans2(con2_2)
            # x2 = torch.relu(convTrans2)
            x2 = -1 * (torch.sigmoid(convTrans2)) + 1

            x2 = self.weight3.mul(x2)
            # enc2 =self.weight4.mul(enc2)
            # x2 = 0.5 * x2
            # enc2 = 0.5 * enc2
            # print(enc2.shape)
            # print(x2.shape)
            # x21 = x2.mul(torch.sigmoid(convTrans2))

            x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
            x2 = x2 + enc2
            # x2 = 1 - x2
            bridge = self.bridge(down3)

            conv1_1 = self.conv1_1(bridge)
            convTrans1 = self.convTrans1(conv1_1)
            # x = torch.relu(convTrans1)
            x = -1 * (torch.sigmoid(convTrans1)) + 1

            # x = 0.5 * x
            # enc3 = 0.5 * enc3
            # print(x.shape)
            # print(enc3.shape)
            # print(x.shape)
            x = self.weight5.mul(x)
            # enc3 = self.weight6.mul(enc3)


            # x11 = x.mul(torch.sigmoid(convTrans1))

            x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
            x = x + enc3
            # x = 1 - x
            up3 = self.up3(bridge)


            up3 = torch.cat((up3, x), dim=1)
            # print(up3.shape)
            dec3 = self.decoder3(up3)

            # print(dec3.shape)
            up2 = self.up2(dec3)
            up2 = torch.cat((up2, x2), dim=1)
            dec2 = self.decoder2(up2)

            up1 = self.up1(dec2)
            up1 = torch.cat((up1, x3), dim=1)
            dec1 = self.decoder1(up1)

            final = self.final(dec1)
            final = F.sigmoid(final)
            return final

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False