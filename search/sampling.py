import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
import torch.nn.functional as F

def tsp_sampling(adj_mat, np_points, edge_index_np=None, sparse_graph=False, 
                 parallel_sampling=1, device="cpu", args=None):
  '''
  Output: tours
  Reference: https://github.com/DIMESTeam/DIMES
  '''
  dist_mat = scipy.spatial.distance_matrix(np_points, np_points)
  adj_mat = adj_mat / args.tau
  dist_mat = torch.from_numpy(dist_mat).unsqueeze(0)
  adj_mat = torch.from_numpy(adj_mat)
  batch_size, n_nodes, _ = dist_mat.shape
  n_samples = args.n_samples

  zex = adj_mat.expand((n_samples, batch_size, n_nodes, n_nodes)).to(device)
  adj_flat = dist_mat.view(batch_size, n_nodes * n_nodes).expand((n_samples, batch_size, n_nodes * n_nodes)).to(device)
  idx = torch.arange(n_nodes).expand((n_samples, batch_size, n_nodes)).to(device)
  mask = torch.ones((n_samples, batch_size, n_nodes), dtype=torch.bool).to(device)
  maskFalse = torch.zeros((n_samples, batch_size, 1), dtype=torch.bool).to(device)
  v0 = u = torch.zeros((n_samples, batch_size, 1), dtype=torch.long).to(device) # starts from v0:=0
  mask.scatter_(dim=-1, index=u, src=maskFalse).to(device)

  y, tours = [], [u]
  for i in range(1, n_nodes):
      zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((n_samples, batch_size, 1, n_nodes))).squeeze(dim=-2).masked_select(mask.clone()).view(n_samples, batch_size, n_nodes - i)
      pei = F.softmax(zei, dim=-1)
      qei = args.epsilon / (n_nodes - i) + (1. - args.epsilon) * pei
      vi = qei.view(n_samples * batch_size, n_nodes - i).multinomial(num_samples=1, replacement=True).view(n_samples, batch_size, 1)
      v = idx.masked_select(mask).view(n_samples, batch_size, n_nodes - i).gather(dim=-1, index=vi)
      y.append(adj_flat.gather(dim=-1, index = u * n_nodes + v))
      u = v
      tours.append(u)
      mask.scatter_(dim=-1, index=u, src=maskFalse)

  y.append(adj_flat.gather(dim=-1, index=u * n_nodes + v0)) # ends at node v0
  y = torch.cat(y, dim=-1).sum(dim=-1) # (batch_size, n_samples)
  tours.append(v0)
  
  solved_tours = torch.cat(tours, dim=-1).squeeze()
  best_idx = torch.argmin(y, dim=0)
  results = solved_tours[best_idx].tolist()

  return results
