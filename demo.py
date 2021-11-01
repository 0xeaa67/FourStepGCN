# -*- encoding: utf-8 -*-
"""
@Time    : 2021/10/31 23:21
@Email   : colflip@163.com
"""
import dgl
import torch
from dgl.data import CoraGraphDataset

'''
data of loading
'''


def build_graph_test():
    """a demo graph: just for graph test
    """
    src_nodes = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
    dst_nodes = torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
    graph = dgl.graph((src_nodes, dst_nodes))
    # edges weights if edges has else 1
    graph.edata["w"] = torch.ones(graph.num_edges())
    return graph


def build_graph_cora():
    # Default: ~/.dgl/
    data = CoraGraphDataset()
    # graph = data[0]

    return data
