# -*- encoding: utf-8 -*-
"""
@Time    : 2021/11/1 10:07
@Email   : colflip@163.com
"""
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np

import modle
import utils


def init_args():
    parser = argparse.ArgumentParser(description='Graph Parameters Set.')
    parser.add_argument('--gpu', metavar='N', type=int, default=-1,
                        help='an integer for the accumulator')
    parser.add_argument('--batch_size', metavar='N', type=int, default=128,
                        help='an integer for the accumulator')
    parser.add_argument('--num_workers', metavar='N', type=int, default=6,
                        help='an integer for the accumulator')
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument('--dim', metavar='N', type=int, default=128,
                        help='an integer for the accumulator')
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")

    args = parser.parse_args()
    return args


def train(data, adj):
    args = init_args()
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    graph = data[0]
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    # print(graph.nodes())

    features = graph.ndata['feat']
    print(features.shape, adj.shape)
    model = modle.GCN(nfeat=features.shape[1], nhid=args.dim, nclass=7, n_layers=args.n_layers, activation=F.relu,
                      dropout=args.dropout)
    if cuda:
        model.cuda()

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    dur = list()
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, adj)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        acc = utils.evaluate(model, features, adj, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, graph.number_of_edges() / np.mean(dur) / 1000))
        # scheduler.step()

    print()
    acc = utils.evaluate(model, features, adj, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))  # Test accuracy ~0.806 (0.793-0.819) (paper: 0.815)
