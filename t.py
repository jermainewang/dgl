import dgl
from dgl.data import RedditDataset
import torch as th
import numpy as np
from dgl.contrib.sampling import sample_neighbors, NeighborSampler
import time

data = RedditDataset(self_loop=True)
train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
g = dgl.DGLGraph(data.graph, readonly=True)
print('Graph created')

# try run
f1 = sample_neighbors(g, train_nid[0:10], 2)
print('Dry run finished')

#print(train_nid[0:1000])

##################### Test 1: Use the new subgraph sampling API ####################
t = time.time()
for i in range(100):
    seed_nodes = train_nid[i*1000:(i+1)*1000]
    f1 = sample_neighbors(g, seed_nodes, 10)
    u, _ = f1.edges(form='uv')
    f2 = sample_neighbors(g, th.unique(u), 10)
    #print(i, f2.number_of_edges())
print('Time:', time.time() - t)

##################### Test 2: Use the old sampler data loader ####################
sampler = NeighborSampler(
        g, 1000, 10,
        neighbor_type='in',
        shuffle=False,
        num_hops=2,
        seed_nodes=train_nid,
        num_workers=4)

t = time.time()
for i, nf in enumerate(sampler):
    #print(i, len(nf.block_eid(0)))
    if i == 99:
        break
print('Time:', time.time() - t)
