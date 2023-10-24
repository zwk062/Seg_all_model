import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F

def conv3d_bn(x, filters, num_row, num_col,num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    '''
        3D Convolutional layers

        Arguments:
            x {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters
            num_z {int} -- length along z axis in filters
        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''
    x = nn.Conv3d(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = nn.BatchNorm3d(axis=4, scale=False)(x)

    if (activation == None):
        return x

        x = Activation(activation, name=name)(x)
    return x
def MultiResBlock(U, inp, alpha =1.67):
    '''
       MultiRes Block

       Arguments:
           U {int} -- Number of filters in a corrsponding UNet stage
           inp {keras layer} -- input layer

       Returns:
           [keras layer] -- [output layer]
       '''
    W =alpha * U

    shortcut = inp
    shortcut = conv3d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, 1, activation=None,
                         padding='same')

    conv3x3 = conv3d_bn(inp, int(W * 0.167), 3, 3, 3, activation='relu', padding='same')

    conv5x5 = conv3d_bn(conv3x3, int(W * 0.333), 3, 3, 3, activation='relu', padding='same')

    conv7x7 = conv3d_bn(conv5x5, int(W * 0.5), 3, 3, 3, activation='relu', padding='same')

    out = np.concatenate([conv3x3, conv5x5, conv7x7], axis=4)
    out = nn.BatchNorm3d(axis=4)(out)

    out += shortcut
    out = nn.ReLU(out)
    out = nn.BatchNorm3d(axis=4)(out)

    return out
