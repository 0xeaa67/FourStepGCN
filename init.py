# -*- encoding: utf-8 -*-
"""
@Time    : 2021/10/31 23:24
@Email   : colflip@163.com
"""
import torch
import dgl
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp


def get_adj(graph):
    graph = add_self_loop(graph)
    # edges weights if edges has weights else 1
    graph.edata["w"] = torch.ones(graph.num_edges())
    adj = coo_matrix((graph.edata["w"], (graph.edges()[0], graph.edges()[1])),
                     shape=(graph.num_nodes(), graph.num_nodes()))

    #  add symmetric edges
    adj = convert_symmetric(adj, sparse=True)
    # adj normalize and transform matrix to torch tensor type

    # adj = utils.preprocess_adj(adj, is_sparse=True)
    return adj


def add_self_loop(graph):
    # add self loop
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph


def convert_symmetric(X, sparse=True):
    # add symmetric edges
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def init_node_feat(graph):
    # init graph node features
    nfeat_dim = graph.number_of_nodes()

    row = list(range(nfeat_dim))
    col = list(range(nfeat_dim))
    indices = torch.from_numpy(
        np.vstack((row, col)).astype(np.int64))
    values = torch.ones(nfeat_dim)

    features = torch.sparse.FloatTensor(indices, values,
                                        (nfeat_dim, nfeat_dim))
    return features
