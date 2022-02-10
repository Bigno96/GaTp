"""
Multi-Layer Perceptron implementation
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Sequence of stacked Linear module + ReLU activation
    Optional Dropout
    Weight initialization -> He-Normal
    """

    def __init__(self, in_features, out_features, hidden_features=(),
                 learn_bias=True, use_dropout=False, dropout_rate=0.2):
        """
        :param in_features: int, input features
        :param out_features: int, output features
        :param hidden_features: int tuple, (f_0, f_1, ...)
                                vector of INPUT features of hidden layers
                                len = # total layers - 1
                                (default: ())
        :param learn_bias: bool, learn the additive bias (default: True)
        :param use_dropout: bool, apply a Dropout after each activation function, except last (default: False)
        :param dropout_rate: float, ratio of Dropout (default: 0.2)
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

    def forward(self, x):
        x = self.blocks(x)
        return x


def init_mlp_weight(m):
    """
    Initialize weights for MLP Network
    Linear layer -> weights: He-normal, no bias
    :param m: torch.nn.layer
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
