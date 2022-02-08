"""
Utility file for Neural Network models
"""

import torch.nn as nn


def init_res_net_weight(m):
    """
    Initialize weights for Residual Network
    Convolutional layer -> weights: He-normal, no bias
    Batch Norm layer -> weights: 1, bias: 0
    Linear layer -> weights: uniform (default), bias: 0
    :param m: torch.nn.layer
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)
