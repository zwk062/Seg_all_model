import torch.nn as nn
import torch.nn.functional as F
import torch
from .encoder import CSA

csa_block = CSA()

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x1, x2):
        # print(x1.size())
        x1 = self.up(x1)
        # input is CHW
        # print(x1.size())
        # print(x2.size())
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

    
class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv3d(1, channels, kernel_size, padding=  kernel_size // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm3d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm3d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm3d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    

# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=1, n_classes=1):
       super(Model, self).__init__()
       self.pfc = PFC(64)
       self.img_channels = img_channels
       self.n_classes = n_classes
       self.maxpool = nn.MaxPool3d(kernel_size=2)
       self.out_conv = nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0)
       self.up_conv1 = Up()
       self.up_conv2 = Up()
       self.up_conv3 = Up()
       self.up_conv4 = Up()
       self.down1 = csa_block.layer1
       self.down2 = csa_block.layer2
       self.down3 = csa_block.layer3
       self.down4 = csa_block.layer4
       self.up1 = csa_block.layer5
       self.up2 = csa_block.layer6
       self.up3 = csa_block.layer7
       self.up4 = csa_block.layer8

       
    def forward(self, x):
        # print("x的维度", x.size())
        x1 = self.pfc(x)
        # print("x1的维度",x1.size())
        x2 = self.maxpool(x1)

        x3 = self.down1(x2)
        # print("x3的维度", x3.size())
        x4 = self.maxpool(x3)
        # print("x4的维度", x4.size())
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        
        x9 = self.down4(x8)

        x10 = self.up_conv1(x9,x7)

        x11 = self.up1(x10)
   
        x12 = self.up_conv2(x11,x5)    
        x13 = self.up2(x12)
       
        x14 = self.up_conv3(x13,x3)   
        x15 = self.up3(x14)
        
        x16 = self.up_conv4(x15,x1)
        x17 = self.up4(x16)
        
        x18 = self.out_conv(x17)
        
        x19 = F.sigmoid(x18)
        return x19
