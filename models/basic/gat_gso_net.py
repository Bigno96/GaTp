"""
The following code is taken from graphML.py Module for basic GSP and graph machine learning functions,
2018/11/01~2018/07/12
Fernando Gama, fgama@seas.upenn.edu.
Luana Ruiz, rubruiz@seas.upenn.edu.
Kate Tolstaya, eig@seas.upenn.edu

1) Functional
LSIGF: Applies a linear shift-invariant graph filter
learnAttentionGSO: Computes the GSO following the attention mechanism

2) Filtering Layers (nn.Module)
GraphAttentional: Creates a layer using graph attention mechanisms


GatGso returns a Graph Convolution Attentional Network Module with support for dynamically adding GSO
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

import math
import numpy as np

ZERO_TOLERANCE = 1e-9    # values below this number are considered zero
INF_NUMBER = 1e12       # infinity equals this number


class GatGSO(nn.Module):
    """
    Graph Convolution Attention Network with Graph Shift Operator
    Attention Mode -> Key Query
    """
    def __init__(self, in_features, out_features, hidden_features=(),
                 graph_filter_taps=(3,), attention_heads=(1,), attention_concat=True):
        """
        :param in_features: int, input features
        :param out_features: int, output, features
        :param hidden_features: int tuple, (f_0, f_1, ...)
                                vector of INPUT features of hidden layers
                                len = # graph filtering layers - 1
                                (default: ())
        :param graph_filter_taps: int tuple (k_0, k_1, ...)
                                  k_i = number of filter taps at i_th layer
                                  len = number of graph filtering layers
                                  (default: (3,))
        :param attention_heads: int tuple, number of attention heads for each layer
                                (default: (1,))
        :param attention_concat: bool,
                                 if True, the output of the attention_heads attention heads are concatenated
                                          to form the output features,
                                 if False, they are averaged
                                 (default: True)
        """
        # initialize parent
        super().__init__()
        # set parameters
        self.in_features = in_features
        self.out_features = out_features
        self.F = [in_features] + list(hidden_features) + [out_features]     # features vector
        self.L = len(graph_filter_taps)  # number of graph filtering layers
        self.K = graph_filter_taps   # array of number of filter taps, for each layer
        self.P = attention_heads    # number of attention heads
        assert len(self.P) == self.L
        assert len(self.F)-1 == self.L

        self.E = 1  # Number of edge features
        self.bias = True
        self.attention_concat = attention_concat    # attention concatenation bool

        self.S = None   # GSO not yet defined

        # build layers, feeding to Sequential
        self.model = nn.Sequential(*[
            GraphFilterBatchAttentional(G=self.F[i], F=self.F[i+1],     # input and output features for each layer
                                        K=self.K[i], P=self.P[i],   # filter taps, attention heads
                                        E=self.E, bias=self.bias,   # edge features, bias
                                        concatenate=self.attention_concat)
            for i in range(self.L)
        ])

    def set_gso(self, S):
        # We set the GSO on real time, this GSO also depends on time and has
        # B -> batch size
        # E -> edge features
        # N -> agents number
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, x):
        for i in range(self.L):
            self.model[i].add_gso(self.S)  # add GSO for GraphFilter

        # F -> input features, G -> output features
        # B x F x N - > B x G x N,
        out = self.model(x)

        return out


class GraphFilterBatchAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer
    Key and Query Attention mode
        https://arxiv.org/abs/2003.09575

    Initialization:
        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               non-linearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            non-linearity (nn.functional): non-linearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.add_gso(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E=1, bias=True, non_linearity=f.relu, concatenate=True):
        # P: Number of heads
        # GSOs will be added later
        # this combines both weight scalars and weight vectors

        # initialize parent
        super().__init__()

        '''save parameters'''
        self.G = G  # in_features
        self.F = F  # out_features
        self.K = K  # filter_taps
        self.P = P  # attention_heads
        self.E = E  # edge_features
        self.S = None   # no GSO assigned yet
        self.N = 0
        self.aij = None
        self.non_linearity = non_linearity
        self.concatenate = concatenate

        '''create parameters'''
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))

        self.weight_bias = nn.parameter.Parameter(torch.Tensor(P, E, F))
        self.filter_weight = nn.parameter.Parameter(torch.Tensor(P, F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Key and Query mode
        # https://arxiv.org/abs/2003.09575
        self.weight = nn.parameter.Parameter(torch.Tensor(P, E, G, G))
        self.graph_attention_LSIGF_batch = graph_attention_lsigf_batch_key_query

        '''initialize parameters'''
        self.set_parameters()

    def set_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        std_v = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-std_v, std_v)
        self.weight_bias.data.uniform_(0, 0)
        self.mixer.data.uniform_(-std_v, std_v)
        self.filter_weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.uniform_(-std_v, std_v)

    def add_gso(self, S):
        # every S has 4 dimensions
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def return_attention_gso(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # every S has 4 dimensions
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        N = x.shape[2]

        # And now we add the zero padding
        if N < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-N)
                                   .type(x.dtype).to(x.device)
                           ), dim=2)

        # And get the graph attention output
        y, aij = self.graph_attention_LSIGF_batch(h=self.filter_weight, x=x, a=self.mixer,
                                                  W=self.weight, S=self.S, b=self.bias)
        self.aij = aij.detach().cpu().numpy()

        # This output is of size B x P x F x N.
        # Now, we can either concatenate them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the non-linearity
            y = self.non_linearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0), (p=1,f=1), ...
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim=1)    # B x F x N
            # And then we apply the non-linearity
            y = self.non_linearity(y)

        if N < self.N:
            y = torch.index_select(y, 2, torch.arange(N).to(y.device))
        return y

    def extra_repr(self):
        repr_string = "in_features=%d, " % self.G
        repr_string += "out_features=%d, " % self.F
        repr_string += "filter_taps=%d, " % self.K
        repr_string += "attention_heads=%d, " % self.P
        repr_string += "edge_features=%d, " % self.E
        repr_string += "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            repr_string += "GSO stored: number_nodes=%d" % self.N
        else:
            repr_string += "no GSO stored"
        return repr_string


# noinspection DuplicatedCode
def graph_attention_lsigf_batch_key_query(h, x, a, W, S, b=None, negative_slope=0.2):

    K = h.shape[3]  # filter_taps
    F = h.shape[4]  # out_features

    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features

    assert W.shape[0] == P
    assert W.shape[1] == E
    assert W.shape[3] == G
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learn_attention_gso_batch_key_query(x=x, a=a, W=W, S=S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # h: P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)

    # the easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single dimension,
    # we would still be unable to handle the E features this way
    # we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)

    # add the k=0 dimension (B x P x E x K x G x N)
    # and now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N

    # this output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of the learned GSO)
    # now, we need to multiply this by the filter coefficients

    # convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # and z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # and multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N

    # finally, add the bias
    if b is not None:
        y = y + b

    return y, aij


# noinspection DuplicatedCode
def learn_attention_gso_batch_key_query(x, a, W, S, negative_slope=0.2):
    """
    Computes the GSO following the attention mechanism
    https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc

    Multiplicative Attention:
        eij = q * W * k

    G the number of input features,
    F the number of output features,
    E the number of edge features,
    P the number of attention heads,
    Ji the number of nodes in N_{i}, the neighborhood of node i,
    N the number of nodes.

    Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight matrix associated to edge feature e and attention head p,
    and a^{ep} in R^{2F} the mixing vector.
    Let alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for edge feature e and attention head p,
    and let s_{ij}^{e} be the value of feature e of the edge connecting nodes i and j.

    Each element of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x input_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """

    B = x.shape[0]  # batch_size
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    # x = batch_size x input_features x number_nodes -> x = batch_size x number_nodes x input_features
    x_key = x.permute([0, 2, 1]).reshape([B, 1, 1, N, G])
    x_query = x_key.permute([0, 1, 2, 4, 3])
    W = W.reshape([1, P, E, G, G])

    # B x P x E x G x G * B x 1 x 1 x G x M
    # = B x P x E x G x N
    Wx = torch.matmul(W, x_query)

    # B x P x E x N x G * B x 1 x 1 x G x N
    # = B x P x E x N x N
    xWx = torch.matmul(x_key, Wx)
    eij = nn.functional.leaky_relu(xWx, negative_slope=negative_slope)

    # apply the softmax
    # we do not want to consider the places where there are no neighbors,
    # so we need to set them to -infinity so that they will be assigned a zero.

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N])    # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > ZERO_TOLERANCE).type(x.dtype).cuda()   # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * INF_NUMBER
    infinityMask.cuda()
    # compute the softmax plus the -infinity (we first force the places where there is no edge to be zero,
    # and then we add -infinity to them)

    aij_tmp = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)

    return aij_tmp * maskEdges
