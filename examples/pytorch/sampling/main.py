import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import time
import argparse
from dgl.data import RedditDataset

class GCNSampling(nn.Module):

    class NodeUpdate(nn.Module):
        def __init__(self, in_feats, out_feats, activation=None):
            super(NodeUpdate, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats)
            self.activation = activation

        def forward(self, nodes):
            h = nodes.data['h']
            h = self.linear(h)
            if self.activation is not None:
                h = self.activation(h)
            return {'h': h} 

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            NodeUpdate(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                NodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(
            NodeUpdate(n_hidden, n_classes))

    def forward(self, nf):
        h = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_u('h', 'm'), fn.mean('m', 'h'), layer)
            h = nf.layers[i+1].data['h']
        return h

class GraphSAGESampling(nn.Module):

    class NodeUpdate(nn.Module):
        def __init__(self, in_feats, out_feats, activation=None):
            super(NodeUpdate, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats)
            self.activation = activation

        def forward(self, nodes):
            h = nodes.data['h']
            h = self.linear(h)
            if self.activation is not None:
                h = self.activation(h)
            return {'h': h} 

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            NodeUpdate(in_feats * 2, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                NodeUpdate(n_hidden * 2, n_hidden, activation))
        # output layer
        self.layers.append(
            NodeUpdate(n_hidden * 2, n_classes))

    def forward(self, nf):
        h = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_u('h', 'm'), fn.mean('m', 'h_n'), layer)
            h = nf.layers[i+1].data['h']
        return h

def compute_acc(pred, labels):
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, nf, labels):
    model.eval()
    with th.no_grad():
        pred = model(nf)
    model.train()
    return compute_acc(pred, labels)

def run(proc_id, n_gpus, args, devices):
    #th.manual_seed(1234)
    #np.random.seed(1234)
    #th.cuda.manual_seed_all(1234)
    #dgl.random.seed(1234)

    # dropout probability
    dropout = 0.2

    # Setup multi-process
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.set_num_threads(args.num_workers * 2 if args.prefetch else args.num_workers)

    # Prepare data
    data = RedditDataset(self_loop=True)
    train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(data.val_mask)[0])
    features = th.Tensor(data.features)
    # Split train_nid
    train_nid = th.split(train_nid, len(train_nid) // n_gpus)[dev_id]

    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels

    # Construct graph
    g = dgl.DGLGraph(data.graph, readonly=True)
    g.ndata['features'] = features

    # Create sampler
    sampler = dgl.contrib.sampling.NeighborSampler(
        g, args.batch_size, args.fan_out,
        neighbor_type='in',
        shuffle=True,
        num_hops=args.num_layers,
        seed_nodes=train_nid,
        num_workers=args.num_workers)

    if proc_id == 0:
        val_sampler = dgl.contrib.sampling.NeighborSampler(
            g, len(val_nid), 10000,
            neighbor_type='in',
            shuffle=False,
            num_hops=args.num_layers,
            seed_nodes=val_nid,
            num_workers=1)
        # Create validation batch (only on GPU 0)
        val_nf = list(val_sampler)[0]
        val_nf.copy_from_parent()
        val_nf.layers[0].data['features'] = val_nf.layers[0].data['features'].to(0)

    # Define model and optimizer
    model = GCNSampling(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    # Training loop
    avg = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        for step, nf in enumerate(sampler):
            nf.copy_from_parent()
            nf.layers[0].data['features'] =\
                nf.layers[0].data['features'].to(dev_id)
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(dev_id)
            batch_labels = labels[batch_nids].to(dev_id)
            # compute loss
            loss = loss_fcn(pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data,
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus
            if n_gpus > 1:
                th.distributed.barrier()
            optimizer.step()
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f}'.format(
                    epoch, step, loss.item(), acc.item()))
        if n_gpus > 1:
            th.distributed.barrier()
        toc = time.time()
        if proc_id == 0:
            eval_acc = evaluate(model, val_nf, labels[val_nid].to(dev_id))
            print('Epoch Time(s): {:.4f} | Eval Acc {:.4f}'.format(toc - tic, eval_acc))
        if epoch >= 10:
            avg += toc - tic

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('avg time: {}'.format(avg / (epoch - 9)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-workers', type=int, default=1)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--prefetch', action='store_true')
    argparser.add_argument('--log-every', type=int, default=20)
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, args, devices)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, args, devices), nprocs=n_gpus)
