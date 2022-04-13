"""
Multi-Layer Perceptron implementation
"""
import torch

import torch.nn as nn

from typing import Optional, Tuple


class MLP(nn.Module):
    """
    Sequence of stacked Linear module + ReLU activation
    Optional Dropout
    Weight initialization -> He-Normal
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[Tuple[int, ...]] = (),
                 learn_bias: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 append_relu: bool = True):
        """
        :param in_features: input features
        :param out_features: output features
        :param hidden_features: (f_0, f_1, ...), vector of INPUT features of hidden layers
                                len = # total layers - 1
                                (default: ())
        :param learn_bias: if True, learn the additive bias
                           (default: True)
        :param use_dropout: if True, apply a Dropout after each activation function, except last layer
                            (default: False)
        :param dropout_rate: ratio of Dropout
                             (default: 0.2)
        :param append_relu: if True, append at the end a ReLU
                            (default: True)
        """
        # initialize parent
        super().__init__()
        # set parameters
        self.F = [in_features] + list(hidden_features) + [out_features]     # features vector
        self.L = len(self.F)-1        # number of layers

        # append sequence of blocks, with optional dropout
        layers = []
        for i in range(self.L):
            # linear transformation
            layers.append(nn.Linear(in_features=self.F[i], out_features=self.F[i+1], bias=learn_bias))

            # if not last layer
            if i < self.L-1:
                layers.append(nn.ReLU(inplace=True))  # activation function
                if use_dropout:     # if dropout is used
                    layers.append(nn.Dropout(p=dropout_rate))
            # if last layer and relu is appended
            elif append_relu:
                layers.append(nn.ReLU(inplace=True))  # activation function

        # create the model
        self.blocks = nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        return self.blocks(x)
