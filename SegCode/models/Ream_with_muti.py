import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.nn.functional as F


class Reduction_Encoder(nn.Module):
    def __init__(self,input,c1,c2,c3):
        super(Reduction_Encoder, self).__init__()
        #线路为1×1×1、5×5×5（步长为2）的卷积链
        #5*5*5的卷积padding应该是2
        #[64-5+4]/2+1 =32
        self.p1 =nn.Sequential(
            nn.Conv3d(input,c1[0],kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(c1[0],c1[1],kernel_size=(5,5,5),stride=(2,2,2),padding=(2,2,2)),
            nn.ReLU()
        )
        #路径为MaxPooling,1×1×1的路径
        self.p2 =nn.Sequential(
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(input,c2,kernel_size=(1,1,1)),
            nn.ReLU()
        )
        #路径为1×1×1（步长为2）、1×1×5、1×5×1、5×1×1
        self.p3 =nn.Sequential(
            nn.Conv3d(input,c3[0],kernel_size=(1,1,1),stride=(2,2,2)),
            nn.ReLU(),
            nn.Conv3d(c3[0],c3[1],kernel_size=(1,1,5)),
            nn.ReLU(),
            nn.Conv3d(c3[1],c3[2],kernel_size=(1,5,1)),
            nn.ReLU(),
            nn.Conv3d(c3[2],c3[3],kernel_size=(5,1,1)),
            nn.ReLU()
        )
    def forward(self,x):
        p1 =self.p1(x)
        p2 =self.p2(x)
        p3 =self.p3(x)
        return torch.cat((p1,p2,p3),dim=1)

class Reduction_Decoder(nn.Module):
    def __init__(self,input,d0,d1,d2,d3,d4):
        super(Reduction_Decoder, self).__init__()
        self.p1 =nn.Sequential(
            nn.Conv3d(input,d0[0],kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(d0[0],d0[1],kernel_size=(5,5,5)),
            nn.ReLU()
        )

        self.p2 =nn.Sequential(
            nn.Conv3d(input,d1[0],kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(d1[0],d1[1],kernel_size=(1,1,7)),
            nn.ReLU()
        )
        self.p3 =nn.Sequential(
            nn.Conv3d(input, d2[0], kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(d2[0], d2[1], kernel_size=(1, 7, 1)),
            nn.ReLU()
        )
        self.p4 =nn.Sequential(
            nn.Conv3d(input, d3[0], kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(d3[0], d3[1], kernel_size=(1, 1, 7)),
            nn.ReLU()
        )
        self.p5 =nn.Sequential(
            nn.Conv3d(input, d4[0], kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(d4[0], d4[1], kernel_size=(1, 1, 7)),
            nn.ReLU(),
            nn.Conv3d(d4[1], d4[2], kernel_size=(1, 7, 1)),
            nn.ReLU(),
            nn.Conv3d(d4[2], d4[3], kernel_size=(7, 1, 1)),
            nn.ReLU()
        )
    def forward(self,x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        p5 = self.p5(x)
        return torch.cat((p1,p2,p3,p4,p5),dim=1)


class Res_muti(nn.Module):
    def __init__(self):
        super(Res_muti, self).__init__()
        self.encoder1 = Reduction_Encoder(1, (64,64), 64, (64,64,64,64))
        self.encoder2 = Reduction_Encoder(64,(128,128), 128, (128,128,128,128))
        self.encoder3 = Reduction_Encoder(128, (256,256), 256, (256,256,256,128))
        self.bridge = Reduction_Encoder(256, (512,512), 512, (512,512,512,512))
        self.weight1 = nn.Parameter(torch.randn(64,64))
        self.weight2 = nn.Parameter(torch.randn(128,128))
        self.weight3 = nn.Parameter(torch.randn(256,256))
        self.conv1_1 = nn.Conv3d(512, 1, kernel_size=(1,1,1))
        self.conv2_2 = nn.Conv3d(256, 1, kernel_size=(1,1,1))
        self.conv3_3 = nn.Conv3d(128, 1, kernel_size=(1,1,1))

        self.convTrans1 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))
        self.convTrans2 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))
        self.convTrans3 = nn.ConvTranspose3d(1, 1, kernel_size=(2,2,2), stride=(2,2,2))

        self.decoder3 = Reduction_Decoder(512,(256,256),(256,256),(256,256),(256,256),(256,256,256,256))
        self.decoder2 = Reduction_Decoder(256,(128,128),(128,128),(128,128),(256,256),(128,128,128,128))
        self.decoder1 = Reduction_Decoder(128,(64,64),(64,64),(64,64),(64,64),(64,64,64,64))
        self.final = nn.Conv3d(64, 1, kernel_size=(1,1,1), padding=(0,0,0))


    def forward(self,x):
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
