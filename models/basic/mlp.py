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
                 learn_bias: bool = True,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2):
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
        """
        # initialize parent
        super().__init__()
        # set parameters
        self.in_features = in_features
        self.out_features = out_features
        self.F = [in_features] + list(hidden_features) + [out_features]     # features vector
        self.L = len(self.F)-1        # number of layers

        # append sequence of blocks, with optional dropout
        layers = []
        for i in range(self.L):
            # linear transformation
            layers.append(nn.Linear(in_features=self.F[i], out_features=self.F[i+1], bias=learn_bias))
            # if not last layer
            if i < self.L-1:
                layers.append(nn.ReLU(inplace=True))    # activation function
                # if dropout is used
                if use_dropout:
                    layers.append(nn.Dropout(p=dropout_rate))   # dropout

        # create the model
        self.blocks = nn.Sequential(*layers)

        # initialize weights
        self.apply(init_mlp_weight)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.blocks(x)
        return x


# TODO
def init_mlp_weight(m: torch.nn.Module) -> None:
    """
    Initialize weights for MLP Network
    Linear layer -> weights: He-normal, no bias
    :param m: torch.nn.layer
    """
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
