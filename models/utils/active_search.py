import torch
from torch import nn
import torch.nn.functional as F

class ActiveSearch(object):
    def __init__(self, dist_mat, num_steps, num_samples, inner_lr):
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.dist_mat = dist_mat
        self.device = dist_mat.device
        self.inner_lr = inner_lr

    @torch.enable_grad()
    def tsp_sample(self, heat_map, samples=1, epsilon=0.): # epsilon exploration
        batch_size, n_nodes, _ = self.dist_mat.shape
        zex = heat_map.expand((samples, batch_size, n_nodes, n_nodes))
        adj_flat = self.dist_mat.view(batch_size, n_nodes * n_nodes).expand((samples, batch_size, n_nodes * n_nodes))
        idx = torch.arange(n_nodes).expand((samples, batch_size, n_nodes)).to(self.device)
        mask = torch.ones((samples, batch_size, n_nodes), dtype=torch.bool).to(self.device)
        maskFalse = torch.zeros((samples, batch_size, 1), dtype=torch.bool).to(self.device)
        v0 = u = torch.zeros((samples, batch_size, 1), dtype=torch.long).to(self.device) # starts from v0:=0
        mask.scatter_(dim=-1, index=u, src=maskFalse)
        y, logp, logq = [], [], []
        for i in range(1, n_nodes):
            zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((samples, batch_size, 1, n_nodes))).squeeze(dim=-2).masked_select(mask.clone()).view(samples, batch_size, n_nodes - i)
            pei = F.softmax(zei, dim=-1)
            qei = epsilon / (n_nodes - i) + (1. - epsilon) * pei
            vi = qei.view(samples * batch_size, n_nodes - i).multinomial(num_samples=1, replacement=True).view(samples, batch_size, 1)
            logp.append(torch.log(pei.gather(dim=-1, index=vi) + 1e-8))
            logq.append(torch.log(qei.gather(dim=-1, index=vi) + 1e-8))
            v = idx.masked_select(mask).view(samples, batch_size, n_nodes - i).gather(dim=-1, index=vi)
            y.append(adj_flat.gather(dim=-1, index = u * n_nodes + v))
            u = v
            mask.scatter_(dim=-1, index=u, src=maskFalse)
        y.append(adj_flat.gather(dim=-1, index=u * n_nodes + v0)) # ends at node v0
        y = torch.cat(y, dim=-1).sum(dim=-1).T # (batch_size, samples)
        logp = torch.cat(logp, dim=-1).sum(dim=-1).T
        logq = torch.cat(logq, dim=-1).sum(dim=-1).T
        return y, logp, logq # (batch_size, samples)

    @torch.enable_grad()
    def active_search(self, emb0, epsilon=0.):
        batch_size, _, _ = self.dist_mat.shape
        emb = nn.Parameter(emb0.detach().clone(), requires_grad=True)
        inner_opt = torch.optim.AdamW([emb], lr=self.inner_lr, weight_decay=0.)
        y_means = []
        y_bl = torch.zeros((batch_size, 1))
        for t in range(1, self.num_steps + 1):
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