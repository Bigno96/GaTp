"""
Graph utilities functions

Compute adjacency matrix, normalized, to use as GSO for Graph Neural Network
The Adjacency Matrix is normalized using node degree and max eigenvalue
"""

import numpy as np
import scipy.spatial as sc

ZERO_TOLERANCE = 1e-9


def compute_adj_matrix(agent_pos_list, comm_radius=7):
    """
    Compute adjacency matrix of agents
    An edge e_ij between agents a_i and a_j is present iff dist(loc(a_i), loc(a_j)) <= comm_radius
    :param agent_pos_list: np.ndarray, shape = (agent_num, 2), list of agent positions
    :param comm_radius: int, maximum distance for communication between agents
                        (default: 7)
    :return: Normalized adjacency matrix
             shape = (agent_num, agent_num)
    """
    # compute pair-wise Euclidean distance of agents position
    dist_matrix = sc.distance_matrix(x=agent_pos_list, y=agent_pos_list, p=2)

    # set W = d_ij if corresponding dist_matrix element d_ij is s.t. d_ij < comm_radius
    W = dist_matrix < comm_radius

    # return the normalized adj matrix
    return normalize_adjacency(W=W)


def normalize_adjacency(W):
    """
    Compute the degree-normalized adjacency matrix
    Normalize also with max Eigenvalue
    :param W: np.ndarray, adjacency matrix
    :return np.ndarray, degree-normalized adjacency matrix
    """
    # check that the matrix is square
    assert W.shape[0] == W.shape[1]

    # if W is not an all-zero matrix
    if np.any(W):
        # compute the degree vector
        d = np.sum(W, axis=1)
        # get where d is zero
        zero_d = np.nonzero(np.abs(d) < ZERO_TOLERANCE)
        # change 0 -> 1
        d[zero_d] = 1.
        # invert the square root of the degree
        d = 1/np.sqrt(d)
        # change back 1-> 0
        d[zero_d] = 0.
        # build the square root inverse degree matrix
        D = np.diag(d)

        # degree normalized adjacency
        W = D @ W @ D

        # get maximum eigenvalue
        max_eig = max_eigenvalue(matrix=W)

        # return normalized adjacency
        return W / max_eig

    # all zero matrix -> do nothing
    else:
        return W


def max_eigenvalue(matrix):
    """
    :param matrix: np.ndarray
    :return: float, maximum eigenvalue of the matrix
    """
    # check symmetry
    is_symmetric = np.allclose(matrix, np.transpose(matrix, axes=[1, 0]))
    if is_symmetric:
        W = np.linalg.eigvalsh(matrix)
    else:
        W = np.linalg.eigvals(matrix)

    # get real part and compute max
    return np.max(np.real(W), axis=0)
