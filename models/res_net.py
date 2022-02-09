"""
Function for building custom Residual Neural Network

Structure based on original paper:
K. He, X. Zhang, S. Ren and J. Sun,
"Deep residual learning for image recognition."
In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

Implementation based on the following git-hub repo:
https://github.com/FrancescoSaverioZuppichini/ResNet.git
"""

from functools import partial
import torch.nn as nn


class ResNet(nn.Module):
    """
    Class for building a Residual Neural Network
    Encoder -> takes input in and encode its features through several ResNet Layers
    Decoder -> takes encoder output, applies Global Average Pooling and extract output features through a
               fully connected layer
    """

    def __init__(self, in_channels, out_features, *args, **kwargs):
        """
        :param in_channels: int, input planes
        :param out_features: int, number of features of the output array
        :param args: optional positional arguments for ResNet Blocks and Conv layer
        :param kwargs: optional keyword arguments for ResNet Blocks and Conv layer
        """
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(in_features=self.encoder.blocks[-1].blocks[-1].expanded_channels,
                                     out_features=out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Conv2dAuto(nn.Conv2d):
    """
    Conv2d with auto padding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


'''Conv2d with 1x1 Kernel and auto padding'''
conv1x1 = partial(Conv2dAuto, kernel_size=(1, 1), bias=False)


'''Conv2d with 3x3 Kernel and auto padding'''
conv3x3 = partial(Conv2dAuto, kernel_size=(3, 3), bias=False)


'''Conv2d with 3x3 Kernel and auto padding'''
conv7x7 = partial(Conv2dAuto, kernel_size=(7, 7), bias=False)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    """
    :param in_channels: int, input planes
    :param out_channels: int, output planes
    :param conv: Conv layer to use, custom or from torch.nn
    :param args: optional positional arguments for Conv layer
    :param kwargs: optional keyword arguments for Conv layer
    :return: Stacked sequentially:
                Conv layer (of the specified type)
                Batch Norm 2D
    """
    return nn.Sequential(conv(in_channels=in_channels, out_channels=out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))


def activation_func(activation):
    """
    Dynamically select activation function
    :return: ModuleDict with options for activation function
    """
    return nn.ModuleDict(modules={
        'relu': nn.ReLU(inplace=True),    # Rectified Linear Unit
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': nn.SELU(inplace=True)      # Scaled Exponential Linear Unit
    })[activation]


class ResidualBlock(nn.Module):
    """
    Residual Block, super class of other building Blocks for ResNet
    """

    def __init__(self, in_channels, out_channels, activation='relu', expansion=1, down_sampling=(1, 1)):
        """
        :param in_channels: int, input planes
        :param out_channels: int, output planes
        :param activation: string, activation function to select from activation_dict
        :param expansion: int, multiplicative factor of expansion for output channels
                          out_channels = in_channels * expansion
                          Default: 1, out_channels = in_channels
        :param down_sampling: factor of down-sampling to apply to the first Conv layer
                              Perform down-sampling directly by convolutional layers using stride
                              Default: (1, 1), no down-sampling
        """
        super().__init__()
        # save parameters needed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.down_sampling = down_sampling
        # save activation
        self.activation = activation
        self.activation_func = activation_func(activation=activation)
        # layers implemented by subclasses
        self.blocks = nn.Identity()
        # save shortcut for skip connection
        self.shortcut = nn.Sequential(
            conv1x1(in_channels=self.in_channels, out_channels=self.expanded_channels, stride=self.down_sampling),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual
        x = self.activation_func(x)

        return x


class BasicBlock(ResidualBlock):
    """
    Classic building Block of a ResNet, inheriting from ResidualBlock
    Structure:
        - Conv (with optional down-sampling) + Batch Norm
        - Activation function
        - Conv + Batch Norm
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, conv=conv3x3, *args, **kwargs):
        """
        :param in_channels: int, input planes
        :param out_channels: int, output planes
        :param conv: Conv layer to use, custom or from torch.nn
        :param args: optional positional arguments for Conv layer
        :param kwargs: optional keyword arguments for Conv layer
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, expansion=1, *args, **kwargs)
        # save layers
        self.blocks = nn.Sequential(
            # Conv + Batch Norm
            conv_bn(in_channels=in_channels, out_channels=out_channels,
                    conv=conv, stride=self.down_sampling, bias=False),   # down sampling by stride, optional
            # activation function
            activation_func(activation=self.activation),
            # Conv + Batch Norm
            conv_bn(in_channels=out_channels, out_channels=self.expanded_channels,    # channels expansion
                    conv=conv, stride=(1, 1), bias=False)       # no stride
        )


class BottleNeckBlock(ResidualBlock):
    """
    Building Block for deep, low parameters ResNet, inheriting from ResidualBlock
    Structure:
        - 1x1 Conv + Batch Norm
        - Activation function
        - 3x3 Conv (with optional down-sampling) + Batch Norm
        - Activation function
        - 1x1 Conv + Batch Norm, with out_channels expansion
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        :param in_channels: int, input planes
        :param out_channels: int, output planes
        :param args: optional positional arguments for Conv layer
        :param kwargs: optional keyword arguments for Conv layer
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            # 1x1 Conv + Batch Norm
            conv_bn(in_channels=self.in_channels, out_channels=self.out_channels, conv=conv1x1),
            # activation function
            activation_func(activation=self.activation),
            # 3x3 Conv with (optional) down-sampling by meaning of stride + Batch Norm
            conv_bn(in_channels=self.out_channels, out_channels=self.out_channels, conv=conv3x3,
                    stride=self.down_sampling),
            # activation function
            activation_func(activation=self.activation),
            # 1x1 Conv + Batch Norm with expansion
            conv_bn(in_channels=self.out_channels, out_channels=self.expanded_channels, conv=conv1x1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet Layer composed by n Blocks stacked one after the other
    Skip connection between each Block
    """
    def __init__(self, in_channels, out_channels, n=1, block=BasicBlock, *args, **kwargs):
        """
        :param in_channels: int, input planes
        :param out_channels: int, output planes
        :param n: int, number of Blocks that compose the Layer
        :param block: nn.Module, Blocks to build the layer
        :param args: optional positional arguments for Block and Conv layer
        :param kwargs: optional keyword arguments for Block and Conv layer
        """
        super().__init__()
        # perform down-sampling directly by convolutional layers that have a stride of 2
        down_sampling = (2, 2) if in_channels != out_channels else (1, 1)
        # save block structure for forward pass
        self.blocks = nn.Sequential(
            # first block with (optional) down-sampling
            block(in_channels=in_channels, out_channels=out_channels,
                  down_sampling=down_sampling, *args, **kwargs),
            # n-1 remaining blocks with no down-sampling and (optional) expansion
            *[block(in_channels=out_channels*block.expansion, out_channels=out_channels,
                    down_sampling=(1, 1), *args, **kwargs)
              for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet Encoder composed by Layers with increasing features
    First Layer ('gate'):
        Conv with 7x7 Kernel and 2x2 Stride
        Batch Norm
        Activation function
        Max Pooling
    Than k specified Layers, where k = len(blocks_size)
    Each layer has n specified Blocks
    """

    def __init__(self, in_channels, blocks_size=(64, 128, 256, 512), depths=(2, 2, 2, 2),
                 activation='relu', block=BasicBlock, *args, **kwargs):
        """
        :param in_channels: int, input planes
        :param blocks_size: int array/tuple, in_channels for each subsequent Layer
        :param depths: int array/tuple, number of blocks in each subsequent Layer
        :param activation: string, activation function to select from activation_dict
        :param block: nn.Module, Blocks to build the layer
        :param args: optional positional arguments for Block and Conv layer
        :param kwargs: optional keyword arguments for Block and Conv layer
        """
        super().__init__()
        # first layer, fixed
        self.gate = nn.Sequential(
            conv_bn(in_channels=in_channels, out_channels=blocks_size[0], conv=conv7x7,
                    stride=(2, 2)),  # 7x7 kernel, stride = 2
            activation_func(activation=activation),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))     # 3x3 max pooling
        )
        # rest of the custom layers
        in_out_block_sizes = list(zip(blocks_size, blocks_size[1:]))
        self.blocks = nn.ModuleList([
            # first layer, no expansion -> will trigger down-sampling
            ResNetLayer(in_channels=blocks_size[0], out_channels=blocks_size[0],
                        n=depths[0], activation=activation, block=block,       # block type
                        *args, **kwargs),
            # rest of the layer, with optional expansion
            *[ResNetLayer(in_channels=in_channels * block.expansion, out_channels=out_channels,
                          n=n, activation=activation, block=block,     # block type
                          *args, **kwargs)
              for (in_channels, out_channels), n in zip(in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet
    Perform global average pooling and output the features array by using a fully connected layer
    """
    def __init__(self, in_features, out_features):
        """
        :param in_features: int, input planes
        :param out_features: int, features to output
        """
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)       # flatten
        x = self.fc(x)
        return x
