import numpy as np


def distances(matrix, positions):
    """

    :param matrix: NxN FC matrix
    :param positions: 3D electrodes positions
    :return: vector of distances
    """
    N = np.size(matrix, 0)
    matrix_triu_index = np.triu_indices(matrix.shape[0], k=1)  # indexes of the elements of the upper triangle

    edges_index = np.nonzero(matrix)
    dist_matrix = np.zeros((N, N))
    for i in np.arange(0, np.size(edges_index, 1)):
        a = positions[edges_index[0][i]]  # node i position
        b = positions[edges_index[1][i]]  # node j position
        dist = np.linalg.norm(a - b)  # np.sqrt(((i-j)**2).sum())  # pdist([i, j], 'euclidean')
        dist_matrix[edges_index[0][i]][edges_index[1][i]] = dist

    dist_triu = dist_matrix[matrix_triu_index]  # vector with the elements of the triangular matrix
    non_zero_dist_triu = dist_triu[np.nonzero(dist_triu)]

    return non_zero_dist_triu


def distances_matrix(matrix, positions):
    """

    :param matrix: NxN FC matrix
    :param positions: 3D electrodes positions
    :return: NxN matrix of distances
    """
    N = np.size(matrix, 0)
    edges_index = np.nonzero(matrix)
    dist_matrix = np.zeros((N, N))
    for i in np.arange(0, np.size(edges_index, 1)):
        a = positions[edges_index[0][i]]
        b = positions[edges_index[1][i]]
        dist = np.linalg.norm(a - b)
        dist_matrix[edges_index[0][i]][edges_index[1][i]] = dist

    return dist_matrix
