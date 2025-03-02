import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from typing import Union
from ml4co_kit import points_to_distmat, to_tensor


class ML4TSPNARActiveSearch(object):
    def __init__(
        self, 
        points: Union[np.ndarray, torch.Tensor], 
        num_steps: int = 100, 
        num_samples: int = 1000, 
        inner_lr: float = 5e-2,
        device: str = "cpu"
    ):
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.points = to_tensor(points).to(device)
        self.distmat = points_to_distmat(self.points)
        self.device = device
        self.inner_lr = inner_lr

    @torch.enable_grad()
    def tsp_sample(
        self, 
        heatmap: torch.Tensor, 
        samples: int=1, 
        epsilon: float=0.
    ): 
        # epsilon exploration
        batch_size, nodes_num, _ = self.distmat.shape
        heatmap = heatmap.to(self.device)
        zex = heatmap.expand((samples, batch_size, nodes_num, nodes_num))
        adj_flat = self.distmat.view(batch_size, nodes_num * nodes_num)
        adj_flat = adj_flat.expand((samples, batch_size, nodes_num * nodes_num))
        idx = torch.arange(nodes_num).expand((samples, batch_size, nodes_num)).to(self.device)
        mask = torch.ones((samples, batch_size, nodes_num), dtype=torch.bool).to(self.device)
        maskFalse = torch.zeros((samples, batch_size, 1), dtype=torch.bool).to(self.device)
        v0 = u = torch.zeros((samples, batch_size, 1), dtype=torch.long).to(self.device) # starts from v0:=0
        mask.scatter_(dim=-1, index=u, src=maskFalse)

        y, logp, logq = [], [], []
        for i in range(1, nodes_num):
            zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((samples, batch_size, 1, nodes_num)))
            zei = zei.squeeze(dim=-2).masked_select(mask.clone()).view(samples, batch_size, nodes_num - i)
            pei = F.softmax(zei, dim=-1)
            qei = epsilon / (nodes_num - i) + (1. - epsilon) * pei
            vi = qei.view(samples * batch_size, nodes_num - i)
            vi = vi.multinomial(num_samples=1, replacement=True).view(samples, batch_size, 1)
            logp.append(torch.log(pei.gather(dim=-1, index=vi) + 1e-8))
            logq.append(torch.log(qei.gather(dim=-1, index=vi) + 1e-8))
            v = idx.masked_select(mask).view(samples, batch_size, nodes_num - i).gather(dim=-1, index=vi)
            y.append(adj_flat.gather(dim=-1, index = u * nodes_num + v))
            u = v
            mask.scatter_(dim=-1, index=u, src=maskFalse)

        y.append(adj_flat.gather(dim=-1, index=u * nodes_num + v0)) # ends at node v0
        y = torch.cat(y, dim=-1).sum(dim=-1).T # (batch_size, samples)
        logp = torch.cat(logp, dim=-1).sum(dim=-1).T
        logq = torch.cat(logq, dim=-1).sum(dim=-1).T
        return y, logp, logq # (batch_size, samples)

    @torch.enable_grad()
    def active_search(
        self, 
        emb0: Union[np.ndarray, torch.Tensor],
        epsilon: float=0.
    ):
        batch_size, _, _ = self.distmat.shape
        emb0 = to_tensor(emb0).detach().clone()
        emb = nn.Parameter(emb0, requires_grad=True)
        inner_opt = torch.optim.AdamW([emb], lr=self.inner_lr, weight_decay=0.)
        y_means = []
        y_bl = torch.zeros((batch_size, 1))
        for _ in tqdm(range(1, self.num_steps + 1), desc="Active Search"):
            inner_opt.zero_grad()
            heatmap = emb
            y, logp, logq = self.tsp_sample(heatmap, self.num_samples, epsilon)
            y_means.append(y.mean().item())
            y_bl = y.mean(dim=-1, keepdim=True)
            J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim=-1).sum()
            J.requires_grad_(True)
            J.backward()
            inner_opt.step()
            del heatmap, y
        return emb