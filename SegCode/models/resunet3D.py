"""
Deep ResUNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


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


class ResUNet(nn.Module):
    #def __init__(self, classes, channels):
    def __init__(self):
        super(ResUNet, self).__init__()
        self.encoder1 = ResEncoder(1, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.bridge = ResEncoder(256, 512)
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)
        initialize_weights(self)

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
class NoDDBResUNet2line(nn.Module):
    def __init__(self):
        super(NoDDBResUNet2line, self).__init__()
        self.encoder1 = ResEncoder(1, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.bridge = ResEncoder(256, 512)
        #self.dblock = DACblock(512)
        self.conv1 = nn.Conv3d(512, 1, kernel_size=1)
        self.conv2 = nn.Conv3d(256, 1, kernel_size=1)
        self.convTrans = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        bridge = self.bridge(down3) # ResEncoder
        # dac = self.dblock(bridge)#
        conv1 = self.conv1(bridge)
        convTrans1 = self.convTrans(conv1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        up3 = torch.cat((up3, x), dim=1)
        conv2 = self.conv2(enc3)
        convTrans2 = self.convTrans(conv2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
class DDBResUNet3line(nn.Module):
    def __init__(self):
        super(DDBResUNet3line, self).__init__()
        self.encoder1 = ResEncoder(1, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.bridge = ResEncoder(256, 512)
        self.dblock1 = DACblock(512)
        self.dblock2 = DACblock(256)
        self.dblock3 = DACblock(128)

        self.conv1 = nn.Conv3d(512, 1, kernel_size=1)
        self.conv2 = nn.Conv3d(256, 1, kernel_size=1)
        self.conv3 = nn.Conv3d(128, 1, kernel_size=1)
        self.convTrans = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)

        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.down = downsample()
        self.up3 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.up1 = deconv(128, 64)
        self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.encoder1(x)
        down1 = self.down(enc1)

        enc2 = self.encoder2(down1)
        down2 = self.down(enc2)

        enc3 = self.encoder3(down2)
        down3 = self.down(enc3)

        bridge = self.bridge(down3) # ResEncoder
        dac1 = self.dblock1(bridge)#
        conv1 = self.conv1(dac1)
        convTrans1 = self.convTrans(conv1)

        x = -1 * (torch.sigmoid(convTrans1)) + 1
        x = x.expand(-1, 256, -1, -1, -1).mul(enc3)
        x = x + enc3

        up3 = self.up3(bridge)
        up3 = torch.cat((up3, x), dim=1)
        dac2 = self.dblock2(enc3)#

        conv2 = self.conv2(dac2)
        convTrans2 = self.convTrans(conv2)
        x2 = -1 * (torch.sigmoid(convTrans2)) + 1
        x2 = x2.expand(-1, 128, -1, -1, -1).mul(enc2)
        x2 = x2 + enc2

        dec3 = self.decoder3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat((up2, x2), dim=1)
        dac3 = self.dblock3(enc2)  #
        conv3 = self.conv3(dac3)
        convTrans3 = self.convTrans(conv3)
        x3 = -1 * (torch.sigmoid(convTrans3)) + 1
        x3 = x3.expand(-1, 64, -1, -1, -1).mul(enc1)
        x3 = x3 + enc1

        dec2 = self.decoder2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return final
