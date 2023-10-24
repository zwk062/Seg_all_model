import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder(nn.Module):
    def __init__(self, input, c1,  c2, c3, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # 线路为1×1×1、5×5×5（步长为2）的卷积链
        # 5*5*5的卷积padding应该是2
        # [64-5+4]/2+1 =32
        self.p1 = nn.Sequential(
            nn.Conv3d(input, c1[0], kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(c1[0]),
            nn.Sigmoid(),
            nn.Conv3d(c1[0], c1[1], kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
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
        self.p3 = nn.Sequential(
            nn.Conv3d(input, c3, kernel_size=(1, 1, 1)),
            # nn.ReLU()
        )
        self.relu =nn.ReLU()

    def forward(self, x):
        p1 = self.p1(x)
        # p2 = self.p2(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        res = torch.cat((p1, p2), dim=1)
        res = p3 + res
        res = self.relu(res)
        return res

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
        self.res = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            # nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        res =self.res(x)
        out = res + out
        out = self.relu(out)
        return out



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = Encoder(1, (24, 48), (8, 16, 16, 16), 64)
        self.encoder2 = Encoder(64, (48, 96), (16, 32, 32, 32), 128)
        self.encoder3 = Encoder(128, (96, 192), (32, 64, 64, 64), 256)
        self.bridge = Encoder(256, (192, 384), (64, 128, 128, 128), 512)
        self.bridge = Encoder(512,1024)
        self.decoder4 = Encoder(1024, 512)
        self.decoder3 = Encoder(512, 256)
        self.decoder2 = Encoder(256, 128)
        self.decoder1 = Encoder(128, 64)
        self.down = downsample()
        self.up4 = deconv(1024, 512)
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        # for param in self.parameters():
        #     param.requires_grad = False
        self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        enc4 = self.encoder4(down3)
        down4 = self.down(enc4)

        bridge = self.bridge(down4)

        up4 = self.up4(bridge)
        up4 = torch.cat((up4, enc4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.up3(dec4)
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
