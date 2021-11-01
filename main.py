# -*- encoding: utf-8 -*-
"""
@Email   : colflip@163.com
@Time    : 2021/10/31 23:21
a example for gcn of four step
https://zhuanlan.zhihu.com/p/422380707
"""
import init
import demo
import train
import utils

print("setup--")
# data of loading
data = demo.build_graph_cora()
# init
graph = data[0]
adj = init.get_adj(graph)
# calculate
norm_adj = utils.preprocess_adj(adj)
# train
train.train(data, norm_adj)
print("end--")
