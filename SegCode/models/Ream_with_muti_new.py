import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.nn.functional as F


# 最大池化
def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)

#卷积
def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2))


# #权值初始化
# def initialize_weights(*models):
#     for model in models:
#         for m in model.models():
#             #isinstance()判断对象是不是一个已知的类型
#             if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()




class Reduction_Encoder(nn.Module):
    def __init__(self,input,c1,c2,c3,**kwargs):
        super(Reduction_Encoder, self).__init__(**kwargs)
        #线路为1×1×1、5×5×5（步长为2）的卷积链
        #5*5*5的卷积padding应该是2
        #[64-5+4]/2+1 =32
        self.p1 =nn.Sequential(
            nn.Conv3d(input,c1[0],kernel_size=(1,1,1)),
            nn.Sigmoid(),
            nn.Conv3d(c1[0],c1[1],kernel_size=(5,5,5),stride=(1,1,1),padding=(2,2,2)),
            nn.ReLU()
        )
        #路径为MaxPooling,1×1×1的路径
        self.p2 =nn.Sequential(
            nn.Conv3d(input,c2,kernel_size=(1,1,1)),
            nn.ReLU()
        )
        #路径为1×1×1（步长为2、1×1×5、1×5×1、5×1×1
        self.p3 =nn.Sequential(
            nn.Conv3d(input,c3[0],kernel_size=(1,1,1),stride=(1,1,1),padding=(2,2,2)),
            nn.ReLU(),
            nn.Conv3d(c3[0],c3[1],kernel_size=(1,1,5)),
            nn.ReLU(),
            nn.Conv3d(c3[1],c3[2],kernel_size=(1,5,1)),
            nn.ReLU(),
            nn.Conv3d(c3[2],c3[3],kernel_size=(5,1,1)),
            nn.ReLU()

        )

    def forward(self,x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        return  torch.cat((p1, p2, p3), dim=1)



class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out



class Res_muti_new(nn.Module):
    def __init__(self):
        super(Res_muti_new, self).__init__()
        self.encoder1 = Reduction_Encoder(1, (8,16), 16, (16,32,32,32))
        self.encoder2 = Reduction_Encoder(64,(16,32), 32, (32,64,64,64))
        self.encoder3 = Reduction_Encoder(128, (32,64), 64, (64,128,128,128))
        self.bridge = Reduction_Encoder(256, (64,128), 128, (128,256,256,256))

        self.conv1_1 = nn.Conv3d(512, 1, kernel_size=(1,1,1))
        self.conv2_2 = nn.Conv3d(256, 1, kernel_size=(1,1,1))
        self.conv3_3 = nn.Conv3d(128, 1, kernel_size=(1,1,1))

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))


        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, 1, kernel_size=(1,1,1), padding=(0,0,0))
        # initialize_weights(self)

    def forward(self,x):
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
        x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
        # print(x3.shape)
        x3 = x3 + enc1
        # print(x3.shape)
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