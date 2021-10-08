import networkx as nx
from itertools import permutations
from networkx.exception import NetworkXNoPath
import numpy as np


def invert_weights(G):
    weight = 'weight'
    Ginv = nx.create_empty_copy(G)
    for (n1, n2) in G.edges():
        if G[n1][n2][weight] != 0:
            dist = 1 / G[n1][n2][weight]
            Ginv.add_edge(n1, n2, weight=dist)
    return Ginv


def efficiency_weighted(G, u, v, weight):
    try:
        eff = 1 / nx.shortest_path_length(G, u, v, weight='weight')
    except NetworkXNoPath:
        eff = 0
    return eff


def global_efficiency_weighted(G):
    # G = nx.from_numpy_matrix(MI_Adj_alpha_unweighted[:, :, i])
    # G = MI_Adj_alpha_unweighted[:, :, i]
    # n = len(G)
    n = G.number_of_nodes()
    G = invert_weights(G)
    denom = n * (n - 1)
    if denom != 0:
        g_eff = sum(efficiency_weighted(G, u, v, weight='weight') for u, v in permutations(G, 2)) / denom
    else:
        g_eff = 0
    return g_eff


def local_efficiency_weighted(G):
    weight = 'weight'
    if G.is_directed():
        new_graph = nx.DiGraph
    else:
        new_graph = nx.Graph

    efficiencies = dict()
    for node in G:
        temp_G = new_graph()
        temp_G.add_nodes_from(G.neighbors(node))
        for neighbor in G.neighbors(node):
            for (n1, n2) in G.edges(neighbor):
                if (n1 in temp_G) and (n2 in temp_G):
                    temp_G.add_edge(n1, n2)

        if weight is not None:
            for (n1, n2) in temp_G.edges():
                temp_G[n1][n2][weight] = G[n1][n2][weight]

        efficiencies[node] = global_efficiency_weighted(temp_G)
    return efficiencies


def average_local_efficiency(G):
    weight = 'weight'
    eff = local_efficiency_weighted(G)
    total = sum(eff.values())
    N = len(eff)
    return total / N


def mean_dist(matrix):
    matrix_trian_index = np.triu_indices(matrix.shape[0], k=1)
    dist_trian = matrix[matrix_trian_index]
    dist_mean_H = dist_trian[np.nonzero(dist_trian)].mean()

    if np.isnan(dist_mean_H):
        dist_mean_LH = 0  # for nan values
    return dist_mean_H


def channel_idx(ch_names):
    RH_idx = []
    LH_idx = []
    CH_idx = []
    for i in range(len(ch_names)):
        if ch_names[i][-1].isdigit() and int(ch_names[i][-1]) % 2 == 0:
            RH_idx = np.append(RH_idx, i)
        elif ch_names[i][-1].isdigit() and int(ch_names[i][-1]) % 2 != 0:
            LH_idx = np.append(LH_idx, i)
        elif ch_names[i][-1] == 'z':
            CH_idx = np.append(CH_idx, i)
        else:
            sys.exit('Channel name {0} is not recognized'.format(ch_names[i]))
    return RH_idx.astype(int), LH_idx.astype(int), CH_idx.astype(int)


def global_laterality(X, ch_names):
    """"""
    RH_idx, LH_idx, CH_idx = channel_idx(ch_names)
    LH = X[LH_idx, :][:, LH_idx]
    mean_LH = mean_dist(LH)

    RH = X[RH_idx, :][:, RH_idx]
    mean_RH = mean_dist(RH)

    g_lat = mean_LH - mean_RH

    return g_lat


def local_laterality(X, ch_names):
    """
    order LEFT-RIGHT
    :param X:
    :param ch_names:
    :return:
    """
    RH_idx, LH_idx, CH_idx = channel_idx(ch_names)
    # LH_names = [ch_names[index] for index in LH_idx]
    # print('LH: {0}'.format(LH_names))

    RH_idx = np.array([5, 4, 12, 11, 10, 17, 16, 20])
    CH_idx = np.array([3, 3, 9, 9, 9, 15, 15, 19])

    LH = X[LH_idx, :][:, LH_idx]
    RH = X[RH_idx, :][:, RH_idx]
    CH = X[CH_idx, :][:, CH_idx]

    lat_homol = np.zeros(len(LH) + len(RH))
    # for each node
    for j in range(len(RH_idx)):
        # HOMOLOGOUS NODES
        lat_homol[j] = (np.sum(LH[j]) - np.sum(RH[j])) / (np.sum(CH[j]))
        lat_homol[j + len(RH_idx)] = (np.sum(RH[j]) - np.sum(LH[j])) / (np.sum(CH[j]))

    return lat_homol


def closeness_centrality_weighted(G):
    weight = 'weight'
    if G.is_directed():
        new_graph = nx.DiGraph
    else:
        new_graph = nx.Graph

    closeness_centrality = dict()
    for node in G:
        temp_G = new_graph()
        temp_G.add_nodes_from(G.neighbors(node))
        for neighbor in G.neighbors(node):
            for (n1, n2) in G.edges(neighbor):
                if (n1 in temp_G) and (n2 in temp_G):
                    temp_G.add_edge(n1, n2)

        if weight is not None:
            for (n1, n2) in temp_G.edges():
                temp_G[n1][n2][weight] = G[n1][n2][weight]

        temp_G_inv = invert_weights(temp_G)

        denom = sum(nx.shortest_path_length(temp_G_inv, u, v, weight='weight') for u, v in permutations(temp_G, 2))
        if denom != 0:
            closenes_n = 1/denom
        else:
            closenes_n = 0
        closeness_centrality[node] = closenes_n
    return closeness_centrality