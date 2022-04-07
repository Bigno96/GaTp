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
import math
import torch

import torch.nn as nn
import torch.nn.functional as f

from typing import Optional, Tuple, List

ZERO_TOLERANCE = 1e-9   # values below this number are considered zero, consider FP16
INF_NUMBER = 1e12    # infinity equals this number, consider FP16


class GatGSO(nn.Module):
    """
    Graph Convolution Attention Network with Graph Shift Operator
    Attention Mode -> Key Query
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[List[int]] = (),
                 graph_filter_taps: Optional[Tuple[int, ...]] = (3,),
                 attention_heads: Optional[Tuple[int, ...]] = (1,),
                 attention_concat: bool = True):
        """
        :param in_features: input features
        :param out_features: output features
        :param hidden_features: vector of INPUT features of hidden layers, (f_0, f_1, ...)
                                len = # graph filtering layers - 1
                                (default: ())
        :param graph_filter_taps: vector (k_0, k_1, ...)
                                  k_i = number of filter taps at i_th layer
                                  len = number of graph filtering layers
                                  (default: (3,))
        :param attention_heads: number of attention heads for each layer
                                (default: (1,))
        :param attention_concat: if True, the output of the attention_heads attention heads are concatenated
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
        self.K = graph_filter_taps  # number of filter taps, for each layer
        self.P = attention_heads    # number of attention heads
        assert len(self.P) == self.L
        assert len(self.F)-1 == self.L

        self.E = 1  # Number of edge features
        self.bias = True    # if True, use bias
        self.attention_concat = attention_concat    # if True, use attention concatenation

        self.S = None   # GSO not yet defined

        # build layers, feeding to Sequential
        self.model = nn.Sequential(*[
            GraphFilterBatchAttentional(G=self.F[i],    # input features for each layer
                                        F=self.F[i+1],  # output features for each layer
                                        K=self.K[i],    # attention heads
                                        P=self.P[i],    # filter taps
                                        E=self.E,   # edge features
                                        bias=self.bias,
                                        concatenate=self.attention_concat)
            for i in range(self.L)
        ])

    def set_gso(self,
                S: torch.Tensor) -> None:
        """
        Set the GSO on real time, shape = (B, E, N, N) or (B, N, N)
        B -> batch size
        E -> edge features
        N -> agents number
        :param S: GSO to be set
        """
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forwards pass of the model
        :param x: input data, shape = (B, F, N)
                  B -> batch size
                  F -> input features
                  N -> agents number
        :return: output data, shape = (B, G, N)
                 B -> batch size
                 G -> output features
                 N -> agents number
        """
        for i in range(self.L):
            self.model[i].add_gso(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        return self.model(x)


class GraphFilterBatchAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer
    Key and Query Attention mode
        https://arxiv.org/abs/2003.09575
    """

    def __init__(self,
                 G: int,
                 F: int,
                 K: int,
                 P: int,
                 E: int = 1,
                 bias: bool = True,
                 non_linearity: nn.functional = f.relu,
                 concatenate: bool = True):
        """
        :param G: in_features, number of input features on top of each node
        :param F: out_features, number of output features on top of each node
        :param K: filter_taps, number of filter taps (power of the GSO)
        :param P: attention_heads, number of attention_heads
        :param E: edge_features, number of features on top of each edge
                  (default: 1)
        :param bias: if True, include a bias in the LSIGF stage
                     (default: True)
        :param non_linearity: non-linearity applied after features have been updated through attention
                              (default:nn.functional.relu)
        :param concatenate: If True, the output of the attention_heads are concatenated to form the output features
                            If False, they are averaged
                            (default: True)
        """
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
        self.S = None  # no GSO assigned yet
        self.N = 0
        self.non_linearity = non_linearity
        self.concatenate = concatenate

        '''create parameters'''
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))
        self.filter_weight = nn.parameter.Parameter(torch.Tensor(P, F, E, K, G))
        # add bias if optioned
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)

        # Key and Query mode
        # https://arxiv.org/abs/2003.09575
        self.weight = nn.parameter.Parameter(torch.Tensor(P, E, G, G))

        '''initialize parameters'''
        self.set_parameters()

    def set_parameters(self) -> None:
        """
        Set weights and parameters
        Taken from _ConvNd internal initialization of parameters:
        """
        std_v = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-std_v, std_v)
        self.mixer.data.uniform_(-std_v, std_v)
        self.filter_weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.uniform_(-std_v, std_v)

    def add_gso(self,
                S: torch.Tensor
                ) -> None:
        """
        Before applying the filter, we need to define the GSO that we are going to use.
        This allows to change the GSO while using the same filtering coefficients,
        as long as the number of edge features is the same
        The number of nodes can change
        :param S: graph shift operator,
                  shape = (Batch, edge_features, number_nodes, number_nodes)
        """
        # every S has 4 dimensions
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass
        :param x: input data,
                  shape = (batch_size, in_features, number_nodes)
        :return: output data,
                 shape = (batch_size, out_features, number_nodes)
        """
        B = x.shape[0]  # batch size
        F = x.shape[1]  # input features
        N = x.shape[2]  # nodes number

        # add the zero padding
        if N < self.N:
            x = torch.cat((x,
                           torch.zeros(size=(B, F, self.N-N),
                                       dtype=x.dtype,
                                       device=x.device)
                           ), dim=2)

        # get the graph attention output
        y = graph_attention_lsigf_batch_key_query(h=self.filter_weight,
                                                  x=x,
                                                  a=self.mixer,
                                                  W=self.weight,
                                                  S=self.S,
                                                  b=self.bias)

        # output is of size B x P x F x N.
        # we can either concatenate them (inner layers) or average them (outer layer)
        if self.concatenate:
            # if concatenated, first apply the non-linearity
            y = self.non_linearity(y)
            # concatenate: make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0), (p=1,f=1), ...
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # no concatenate -> first average
            y = torch.mean(y, dim=1)    # B x F x N
            # then apply the non-linearity
            y = self.non_linearity(y)

        if N < self.N:
            y = torch.index_select(y, 2, torch.arange(N).to(y.device))
        return y

    def extra_repr(self) -> str:
        repr_string = "in_features=%d, " % self.G
        repr_string += "out_features=%d, " % self.F
        repr_string += "filter_taps=%d, " % self.K
        repr_string += "attention_heads=%d, " % self.P
        repr_string += "edge_features=%d, " % self.E
        repr_string += "bias=%s, " % (self.bias is not None)
        if self.S.shape[0]:
            repr_string += "GSO stored: number_nodes=%d" % self.N
        else:
            repr_string += "no GSO stored"
        return repr_string


# noinspection DuplicatedCode
def graph_attention_lsigf_batch_key_query(h: nn.Parameter,
                                          x: torch.Tensor,
                                          a: nn.Parameter,
                                          W: nn.Parameter,
                                          S: torch.Tensor,
                                          b: nn.Parameter = None,
                                          negative_slope: float = 0.2
                                          ) -> torch.Tensor:
    """
    Compute graph attention with key_query
    :param h: filter weights,
              shape = (attention_heads, out_features, edge_features, filter_taps, in_features)
    :param x: input data,
              shape = (batch_size, in_features, number_nodes)
    :param a: mixer,
              shape = (attention_heads, edge_features, 2 * out_features)
    :param W: attention weights,
              shape = (heads_number, edge_features, input_features)
    :param S: GSO,
              shape = (batch, edge_features, number_nodes, number_nodes)
    :param b: bias,
              shape = (out_features, 1)
    :param negative_slope: slope of the leaky ReLu,
                           (default: 0.2)
    :return: GSO output
    """
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

    # B x P x E x N x N
    aij = learn_attention_gso_batch_key_query(x=x,
                                              a=a,
                                              W=W,
                                              S=S,
                                              negative_slope=negative_slope)

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

    return y


# noinspection DuplicatedCode
def learn_attention_gso_batch_key_query(x: torch.Tensor,
                                        a: nn.Parameter,
                                        W: nn.Parameter,
                                        S: torch.Tensor,
                                        negative_slope: float = 0.2
                                        ) -> torch.Tensor:
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

    :param x: input,
               shape = (batch_size, input_features, number_nodes)
    :param a: mixing parameter,
              shape = (number_heads, edge_features, 2 * output_features)
    :param W: linear parameter,
              shape = (number_heads, edge_features, input_features, input_features)
    :param S: graph shift operator,
              shape = (batch_size, edge_features, number_nodes, number_nodes)
    :param negative_slope: slope of the leaky ReLu
    :return output GSO,
            shape = (batch_size, number_heads, edge_features, number_nodes, number_nodes)
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

    # first, get places where we have edges
    mask_edges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N])    # B x 1 x 1 x N x N
    # make it a binary matrix
    mask_edges = (mask_edges > ZERO_TOLERANCE).type(x.dtype).to(eij.device)   # B x 1 x 1 x N x N
    # make it -infinity where there are zeros
    infinity_mask = (1 - mask_edges) * INF_NUMBER
    infinity_mask = infinity_mask.to(eij.device)

    # compute the softmax plus the -infinity (we first force the places where there is no edge to be zero,
    # and then we add -infinity to them)
    aij_tmp = nn.functional.softmax(eij * mask_edges - infinity_mask, dim=4)

    return aij_tmp * mask_edges
