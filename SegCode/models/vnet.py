import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
    def forward(self, x):
        pass
    
class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = activation_func()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.

    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_1 + z_3


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv
        return self.conv_x3(rhs_add)


def crop(large, small):
    """large / small with shape [batch_size, channels, depth, height, width]"""

    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        activation_func())


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        activation_func()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels, criterion):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)#[2, 2, 64, 64, 64]
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)#[2, 2, 64, 64, 64]
        if criterion == 'nll':
            self.softmax = F.log_softmax
        else:
            assert criterion == 'dice', "Expect `dice` (dice loss) or `nll` (negative log likelihood loss)."
            self.softmax = F.softmax

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))#[2, 2, 64, 64, 64]
        # Put channel axis in the last dim for softmax.
        y_perm = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        y_flat = y_perm.view(-1, 2)
        return self.softmax(y_flat)


class VNet(nn.Module):
    #def __init__(self, criterion):
    def __init__(self):
        super(VNet, self).__init__()
        self.conv_1 = conv3d_x3(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x3(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        self.bottom = conv3d_x3(256, 256)

        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x3(128, 64)
        self.deconv_1 = deconv3d_x3(64, 32)
        self.final = nn.Conv3d(32, 1, kernel_size=1, padding=0)

        #self.out = softmax_out(32, 2, criterion)

    def forward(self, x):
        conv_1 = self.conv_1(x)#[2, 16, 64, 64, 64]
        pool = self.pool_1(conv_1)#[2, 32, 32, 32, 32]
        conv_2 = self.conv_2(pool)#[2, 32, 32, 32, 32]
        pool = self.pool_2(conv_2)#[2, 64, 16, 16, 16]
        conv_3 = self.conv_3(pool)#[2, 64, 16, 16, 16]
        pool = self.pool_3(conv_3)#[2, 128, 8, 8, 8]
        conv_4 = self.conv_4(pool)#[2, 128, 8, 8, 8])
        pool = self.pool_4(conv_4)#[2, 256, 4, 4, 4]
        bottom = self.bottom(pool)#[2, 256, 4, 4, 4]
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)#[2, 32, 64, 64, 64]
        final = self.final(deconv)
        final = F.sigmoid(final)
        #print(1)
        #return self.out(deconv)#[524288, 2]
        return final
