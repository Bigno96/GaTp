"""
2018/12/03~2018/07/12
Fernando Gama, fgama@seas.upenn.edu.
Luana Ruiz, rubruiz@seas.upenn.edu.

Taken from graphTools.py: Tools for handling graphs

Functions:
    - normalize_adjacency
"""

import numpy as np
import scipy.spatial as sc


def compute_adj_matrix(agent_pos_list, comm_radius=7, edge_weight=1):
    """
    Compute adjacency matrix of agents
    An edge e_ij between agents a_i and a_j is present iff dist(loc(a_i), loc(a_j)) <= comm_radius
    :param agent_pos_list: np.ndarray, shape = (agent_num, 2), list of agent positions
    :param comm_radius: int, maximum distance for communication between agents
                        (default: 7)
    :param edge_weight: int, weight of each edge
                        (default: 1)
    :return: Normalized adjacency matrix
             shape = (agent_num, agent_num)
    """
    # compute pair-wise Euclidean distance of agents position
    dist_matrix = sc.distance_matrix(x=agent_pos_list, y=agent_pos_list, p=2)

    # init adj matrix, shape = (agent_num, agent_num)
    W = np.zeros_like(dist_matrix)
    # set W = edge_weight if corresponding dist_matrix element d_ij is s.t. 0 < d_ij <= comm_radius
    # is strictly greater than 0 to avoid self loops (agent can be at distance 0 only with himself)
    W[np.nonzero((0 < dist_matrix) & (dist_matrix <= comm_radius))] = edge_weight

    # return the normalized adj matrix
    return normalize_adjacency(W=W)


def normalize_adjacency(W):
    """
    Compute the degree-normalized adjacency matrix

    Input:
        W (np.array): adjacency matrix

    Output:
        A (np.array): degree-normalized adjacency matrix
    """
    # check that the matrix is square
    assert W.shape[0] == W.shape[1]

    with np.errstate(divide='ignore', invalid='ignore'):
        # compute the degree vector
        d = np.sum(W, axis=1)
        # invert the square root of the degree
        d = 1/np.sqrt(d)
        # change nan and inf values
        np.nan_to_num(d, copy=False)
        # build the square root inverse degree matrix
        D = np.diag(d)

    # return the Normalized Adjacency
    return D @ W @ D
