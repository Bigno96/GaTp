"""
Function for building custom Residual Neural Network

Structure based on original paper:
K. He, X. Zhang, S. Ren and J. Sun,
"Deep residual learning for image recognition."
In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

Implementation based on the following git-hub repo:
https://github.com/FrancescoSaverioZuppichini/ResNet.git
"""
import torch
import math

import torch.nn as nn

from functools import partial
from typing import Any, Type, Tuple


class ResNet(nn.Module):
    """
    Class for building a Residual Neural Network
    Encoder -> takes input in and encode its features through several ResNet Layers
    Decoder -> takes encoder output, applies Global Average Pooling and extract output features through a
               fully connected layer
    """

    def __init__(self,
                 in_channels: int,
                 out_features: int,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.2,
                 expansion: int = 1,
                 flatten_correction: int = 1,
                 *args: Any,
                 **kwargs: Any):
        """
        :param in_channels: input planes
        :param decoder_in_features: number of features after the avg pool layer
        :param out_features: number of features of the output array
        :param use_dropout: if True, add Dropout
                            (default: True)
        :param dropout_rate: ratio of Dropout when used
                             (default: 0.2)
        :param expansion: intermediate feature expansion factor
                          (default: 1)
        :param flatten_correction: factor for correcting number of features after Flatten layer
                                   (default: 1)
        :param args: optional positional arguments for ResNet Blocks and Conv layer
        :param kwargs: optional keyword arguments for ResNet Blocks and Conv layer
        """
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(in_features=self.encoder.blocks[-1].blocks[-1].get_output_features,
                                     out_features=out_features,
                                     use_dropout=use_dropout,
                                     dropout_rate=dropout_rate,
                                     expansion=expansion,
                                     flatten_correction=flatten_correction)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class Conv2dAuto(nn.Conv2d):
    """
    Conv2d with auto padding
    """
    def __init__(self,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


'''Conv2d with 1x1 Kernel and auto padding'''
conv1x1 = partial(Conv2dAuto, kernel_size=(1, 1), bias=False)


'''Conv2d with 3x3 Kernel and auto padding'''
conv3x3 = partial(Conv2dAuto, kernel_size=(3, 3), bias=False)


def conv_bn(in_channels: int,
            out_channels: int,
            conv: Type[nn.Conv2d] or partial[nn.Conv2d],
            *args: Any,
            **kwargs: Any
            ) -> nn.Sequential:
    """
    :param in_channels: input planes
    :param out_channels: output planes
    :param conv: Conv layer to use, custom or from torch.nn
    :param args: optional positional arguments for Conv layer
    :param kwargs: optional keyword arguments for Conv layer
    :return: Stacked sequentially:
                Conv layer (of the specified type)
                Batch Norm 2D
    """
    return nn.Sequential(conv(in_channels=in_channels,
                              out_channels=out_channels,
                              *args,
                              **kwargs),
                         nn.BatchNorm2d(out_channels))


def activation_func(activation: str) -> nn.Module:
    """
    Dynamically select activation function
    :param activation: 'relu', 'leaky_relu' or 'selu'
    :return: Module with selected activation function
    """
    return nn.ModuleDict(modules={
        'relu': nn.ReLU(inplace=True),  # Rectified Linear Unit
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': nn.SELU(inplace=True)   # Scaled Exponential Linear Unit
    })[activation]


class ResidualBlock(nn.Module):
    """
    Residual Block, building Block for ResNet
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv: Type[nn.Conv2d] or partial[nn.Conv2d] = conv3x3,
                 activation: str = 'relu',
                 down_sampling: Tuple[int, int] = (1, 1)):
        """
        :param in_channels: input planes
        :param out_channels: output planes
        :param activation: activation function to select from activation_dict
                           options: 'relu', 'leaky_relu', 'selu'
        :param down_sampling: factor of down-sampling to apply to the first Conv layer and the residual connection
                              Perform down-sampling directly by convolutional layers using stride
                              (Default: (1, 1), no down-sampling)
        """
        super().__init__()
        self.out_channels = out_channels
        self.activation_func = activation_func(activation=activation)

        # shortcut for skip connection, Conv + Batch Norm
        self.shortcut = conv_bn(in_channels=in_channels,
                                out_channels=out_channels,
                                conv=conv1x1,
                                stride=down_sampling,   # down sampling by stride, optional
                                bias=False)

        # layers
        self.blocks = nn.Sequential(
            # Conv + Batch Norm
            conv_bn(in_channels=in_channels,
                    out_channels=out_channels,
                    conv=conv,
                    stride=down_sampling,  # down sampling by stride, optional
                    bias=False),
            # activation function
            activation_func(activation=activation),
            # Conv + Batch Norm
            conv_bn(in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    stride=(1, 1),  # no stride always
                    bias=False)
        )

    @property
    def get_output_features(self) -> int:
        """
        :return: output features of the layer
        """
        return self.out_channels

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        # apply skip connection
        residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual   # sum back skip connection
        return self.activation_func(x)


class ResNetLayer(nn.Module):
    """
    A ResNet Layer composed by n Blocks stacked one after the other
    Skip connection at each Block
    Down sampling, when active, is used for first conv layer of the block and for the residual shortcut
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 block: Type[ResidualBlock] = ResidualBlock,
                 use_down_sampling: bool = True,
                 *args: Any,
                 **kwargs: Any):
        """
        :param in_channels: input planes
        :param out_channels: output planes
        :param n: number of Blocks that compose the Layer
        :param block: Blocks to build the layer
        :param use_down_sampling: if True, down sampling is used in the first block of the layer
        :param args: optional positional arguments for Block and Conv layer
        :param kwargs: optional keyword arguments for Block and Conv layer
        """
        super().__init__()
        # perform down-sampling directly by convolutional layers that have a stride of 2
        if use_down_sampling:
            down_sampling = (2, 2) if in_channels != out_channels else (1, 1)
        else:
            down_sampling = (1, 1)
        # save block structure for forward pass
        self.blocks = nn.Sequential(
            # first block with (optional) down-sampling
            block(in_channels=in_channels,
                  out_channels=out_channels,
                  down_sampling=down_sampling,
                  *args,
                  **kwargs),
            # n-1 remaining blocks with no down-sampling
            *[block(in_channels=out_channels,
                    out_channels=out_channels,
                    down_sampling=(1, 1),
                    *args,
                    **kwargs)
              for _ in range(n - 1)]
        )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        return self.blocks(x)


class ResNetEncoder(nn.Module):
    """
    ResNet Encoder composed by Layers with increasing features
    First Layer ('gate'):
        Conv with 3x3 Kernel
        Batch Norm
        Activation function
    Than k specified Layers, where k = len(blocks_size)
    Each layer has n specified Blocks
    """

    def __init__(self,
                 in_channels: int,
                 blocks_size: Tuple[int, ...] = (32, 64, 128, 256),
                 depths: Tuple[int, ...] = (2, 2, 2, 2),
                 activation: str = 'relu',
                 block: Type[ResidualBlock] = ResidualBlock,
                 use_down_sampling: bool = True,
                 *args: Any,
                 **kwargs: Any):
        """
        :param in_channels: input planes
        :param blocks_size: in_channels for each subsequent Layer
        :param depths: number of blocks in each subsequent Layer
        :param activation: activation function to select from activation_dict
                           options: 'relu', 'leaky_relu', 'selu'
        :param block: Blocks to build the layer
        :param use_down_sampling: if True, down sampling is used in the first block of the layer
        :param args: optional positional arguments for Block and Conv layer
        :param kwargs: optional keyword arguments for Block and Conv layer
        """
        super().__init__()
        # first layer, fixed
        self.gate = nn.Sequential(
            conv_bn(in_channels=in_channels,
                    out_channels=blocks_size[0],
                    conv=conv3x3,
                    stride=(1, 1)),    # 3x3 kernel, stride = 1
            activation_func(activation=activation)
        )
        # rest of the custom layers
        in_out_block_sizes = list(zip(blocks_size, blocks_size[1:]))
        self.blocks = nn.ModuleList([
            # first layer
            ResNetLayer(in_channels=blocks_size[0],
                        out_channels=blocks_size[0],
                        n=depths[0],
                        activation=activation,
                        block=block,    # block type
                        use_down_sampling=use_down_sampling,
                        *args,
                        **kwargs),
            # rest of the layer
            *[ResNetLayer(in_channels=inp_channels,
                          out_channels=out_channels,
                          n=n,
                          activation=activation,
                          block=block,     # block type
                          use_down_sampling=use_down_sampling,
                          *args,
                          **kwargs)
              for (inp_channels, out_channels), n in zip(in_out_block_sizes, depths[1:])]
        ])

        # initialize weights
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet
    Perform global average pooling and output the features array by using a fully connected layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.2,
                 expansion: int = 1,
                 flatten_correction: int = 1):
        """
        :param in_features: input planes
        :param out_features: features to output
        :param use_dropout: if True, add Dropout
        :param dropout_rate: ratio of Dropout when used
        :param expansion: intermediate feature expansion factor
        :param flatten_correction: factor for correcting number of features after Flatten layer
        """
        super().__init__()
        # save dropout bool
        self.use_dropout = use_dropout

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)   # global average pool with a stride of 2
        self.conv = conv1x1(in_channels=in_features,
                            out_channels=int(in_features * expansion),
                            bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=int(in_features * expansion * flatten_correction),
                            out_features=out_features,
                            bias=True)
        self.activation = nn.ReLU(inplace=True)     # always ReLu

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.pool(x)
        x = self.conv(x)

        # if dropout used
        if self.use_dropout:
            x = self.dropout(x)

        x = self.flat(x)
        x = self.fc(x)

        return self.activation(x)


def init_weights(m: nn.Module) -> None:
    """
    :param m: module to init
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
