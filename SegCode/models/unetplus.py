"""
Deep ResUNet
"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def downsample():
#     return nn.MaxPool3d(kernel_size=2, stride=2)


# def deconv(in_channels, out_channels):
#     return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


# def initialize_weights(*models):
#     for model in models:
#         for m in model.modules():
#             if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=False)
#         # self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         # residual = self.conv1x1(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         # out += residual
#         # out = self.relu(out)
#         return out


# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         out = self.conv(x)
#         return out


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.encoder1 = Encoder(1, 64)
#         self.encoder2 = Encoder(64, 128)
#         self.encoder3 = Encoder(128, 256)
#         self.encoder4 = Encoder(256, 512)
#         self.bridge = Encoder(512,1024)
#         self.decoder4 = Encoder(1024, 512)
#         self.decoder3 = Encoder(512, 256)
#         self.decoder2 = Encoder(256, 128)
#         self.decoder1 = Encoder(128, 64)
#         self.down = downsample()
#         self.up4 = deconv(1024, 512)
#         self.up3 = deconv(512, 256)
#         self.up2 = deconv(256, 128)
#         self.up1 = deconv(128, 64)
#         self.final = nn.Conv3d(64, 1, kernel_size=1, padding=0)
#         initialize_weights(self)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         down1 = self.down(enc1)

#         enc2 = self.encoder2(down1)
#         down2 = self.down(enc2)

#         enc3 = self.encoder3(down2)
#         down3 = self.down(enc3)

#         enc4 = self.encoder4(down3)
#         down4 = self.down(enc4)

#         bridge = self.bridge(down4)

#         up4 = self.up4(bridge)
#         up4 = torch.cat((up4, enc4), dim=1)
#         dec4 = self.decoder4(up4)

#         up3 = self.up3(dec4)
#         up3 = torch.cat((up3, enc3), dim=1)
#         dec3 = self.decoder3(up3)

#         up2 = self.up2(dec3)
#         up2 = torch.cat((up2, enc2), dim=1)
#         dec2 = self.decoder2(up2)

#         up1 = self.up1(dec2)
#         up1 = torch.cat((up1, enc1), dim=1)
#         dec1 = self.decoder1(up1)

#         final = self.final(dec1)
#         final = F.sigmoid(final)
#         return final
        #######################################
#from UNetfamily.layers import _init_paths
import torch
import torch.nn as nn
from models.UNetfamily.layers import unetConv2, unetUp
from models.UNetfamily.utils import init_weights, count_param

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=1, n_classes=1, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=False):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool(X_00)    # 16*256*256
        # print(X_00.shape)
        X_10= self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = [final_1,final_2,final_3,final_4]

        if self.is_ds:
            return final
        else:
            return final_4

# if __name__ == '__main__':
#     print('#### Test Case ###')
#     from torch.autograd import Variable
#     x = Variable(torch.rand(2,1,64,64)).cuda()
#     model = UNet_Nested().cuda()
#     param = count_param(model)
#     y = model(x)
#     print('Output shape:',y.shape)
#     print('UNet++ totoal parameters: %.2fM (%d)'%(param/1e6,param))