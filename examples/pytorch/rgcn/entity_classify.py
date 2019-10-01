"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from dgl.contrib.data import load_data
from dgl.data.rdf import AIFB, MUTAG, BGS, AM
import dgl.function as fn
from functools import partial

from model import BaseRGCN
from aminer import AMINER

class RelGraphConv1(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv1, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        #if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
        if self.num_bases is None or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            #if self.num_bases < self.num_rels:
                # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            #if self.num_bases < self.num_rels:
                #nn.init.xavier_uniform_(self.w_comp,
                                        #gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(torch.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == torch.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        """ Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        torch.Tensor
            New node features.
        """
        g = g.local_var()
        #g.ndata['h'] = x
        #g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight)
        # message passing
        weight = self.weight.view(self.num_bases,
                                  self.in_feat * self.out_feat)
        weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels, self.in_feat, self.out_feat)

        x = x.view(1, x.shape[0], x.shape[1])
        h = torch.matmul(x, weight).squeeze()
        hs = h.split(1)
        for i in range(len(hs)):
            g.ndata['h'] = hs[i].squeeze()
            e = torch.nonzero(etypes == i).squeeze()
            g.send_and_recv(e, fn.copy_u('h', 'm'), fn.sum('m', 'h%d' % i))
        g.ndata['h'] = torch.cat([g.ndata['h%d' % i].unsqueeze(0) for i in range(len(hs))]).sum(0)
        #g.update_all(self.message_func, fn.sum(msg='msg', out='h'))

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr

class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv1(self.in_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv1(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv1(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, dim=1),
                self_loop=self.use_self_loop)

def main(args):
    # load graph data
    #data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    if args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
        if args.dataset == 'am':
            data = AM()
        elif args.dataset == 'aifb':
            data = AIFB()
        elif args.dataset == 'mutag':
            data = MUTAG()
        elif args.dataset == 'bgs':
            data = BGS()
        g = dgl.to_homo(data.graph)
        num_nodes = g.number_of_nodes()
        edge_type = g.edata[dgl.ETYPE]
        num_rels = int(edge_type.max()) + 1
        num_classes = data.num_classes
        labels = torch.zeros((num_nodes,)).long()
        train_idx = data.train_idx
        test_idx = data.test_idx
    elif args.dataset == 'aminer':
        hg = AMINER()
        g = dgl.to_homo(hg)
        num_nodes = g.number_of_nodes()
        edge_type = g.edata[dgl.ETYPE]
        num_rels = int(edge_type.max()) + 1
        num_classes = 8
        labels = torch.randint(0, num_classes, (num_nodes,)).long()
        train_idx = torch.tensor(np.random.permutation(np.arange(num_nodes))[0:1000])
        test_idx = torch.tensor(np.random.permutation(np.arange(num_nodes))[0:200])
    else:
        data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
        num_nodes = data.num_nodes
        num_rels = data.num_rels
        num_classes = data.num_classes
        labels = data.labels
        train_idx = data.train_idx
        test_idx = data.test_idx
        edge_type = torch.from_numpy(data.edge_type)
        labels = torch.from_numpy(labels).view(-1)
        # create graph
        g = DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(data.edge_src, data.edge_dst)

    #assert False

    print('#nodes:', num_nodes)
    print('#rel:', num_rels)
    print('#class:', num_classes)
    print('#train samples:', len(train_idx))
    print('#test samples:', len(test_idx))
    print('#labels:', len(labels))

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    #feats = torch.arange(num_nodes)
    IN_DIM = 100
    feats = torch.randn((num_nodes, IN_DIM))

    # edge type and normalization factor
    #edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)
    edge_norm = None

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        #edge_norm = edge_norm.cuda()
        labels = labels.cuda()

    # create model
    model = EntityClassify(IN_DIM,
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch >= 3:
            t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        print(loss.item())
        if epoch >= 3:
            t1 = time.time()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            t2 = time.time()

        if epoch >= 3:
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t0)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, np.average(forward_time), np.average(backward_time)))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    #args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    args.bfs_level = 0
    main(args)
