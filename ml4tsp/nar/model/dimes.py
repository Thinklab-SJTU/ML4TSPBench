import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple, Sequence
from ml4tsp.nar.env import ML4TSPNAREnv
from ml4tsp.nar.decoder import ML4TSPNARDecoder
from ml4tsp.nar.model.base import ML4TSPNARBaseModel
from ml4tsp.nar.local_search import ML4TSPNARLocalSearch
from ml4tsp.nar.encoder.gnn.gnn_encoder import GNNEncoder


class ML4TSPDIMES(ML4TSPNARBaseModel):
    def __init__(
        self,
        env: ML4TSPNAREnv,
        encoder: GNNEncoder,
        decoder: Union[ML4TSPNARDecoder, str] = "greedy",
        local_search: Union[ML4TSPNARLocalSearch, str] = None,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        # super
        super(ML4TSPDIMES, self).__init__(
            env=env,
            encoder=encoder,
            decoder=decoder,
            local_search=local_search,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            pretrained_path=pretrained_path
        )

    def tsp_inner_loop(
        self, graph: torch.Tensor, emb0: torch.Tensor, steps: int, samples: int, epsilon: float=0.
    ) -> Sequence[Tensor]:
        batch_size = graph.shape[0]
        emb = nn.Parameter(emb0.detach().clone(), requires_grad=True)
        inner_opt = torch.optim.AdamW([emb], lr=self.inner_lr, weight_decay=0.)
        y_means = []
        y_bl = torch.zeros((batch_size, 1))
        for _ in range(1, steps + 1): #tbar:
            inner_opt.zero_grad()
            heatmap = emb
            y, logp, logq = self.tsp_sample(graph, heatmap, 'softmax', samples, epsilon)
            y_means.append(y.mean().item())
            y_bl = y.mean(dim=-1, keepdim=True)
            J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim=-1).sum()
            J.requires_grad_(True)
            J.backward()
            inner_opt.step()
        return emb, y_means
    

    def tsp_outer_loop(
        self, points: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        if self.env.mode == "train":
            opt = self.optimizers()
            opt.zero_grad()
            par0 = self.forward(points, graph, edge_index=None)
            par0 = par0.squeeze(dim=1)
            loss = self.tsp_sample(graph, par0, "greedy")[0].mean().item()
            par1, _ = self.tsp_inner_loop(graph, par0, self.inner_epochs, self.inner_samples)
            par0.backward(par1.grad / graph.shape[0])
            opt.step()
        else:
            par0 = self.forward(points, graph, edge_index=None)
            par0 = par0.squeeze(dim=1)
            loss = self.tsp_sample(graph, par0, "greedy")[0].mean().item()
        return loss

    def tsp_sample(
        self, graph: torch.Tensor, ze: torch.Tensor,
        mode: str = "softmax", samples: int=1, epsilon: float=0.
    ) -> Sequence[Tensor]: 
        # epsilon exploration
        device = graph.device
        batch_size, nodes_num, _ = graph.shape
        zex = ze.expand((samples, batch_size, nodes_num, nodes_num))
        adj_flat = graph.view(batch_size, nodes_num * nodes_num)
        adj_flat = adj_flat.expand((samples, batch_size, nodes_num * nodes_num))
        idx = torch.arange(nodes_num).expand((samples, batch_size, nodes_num)).to(device)
        mask = torch.ones((samples, batch_size, nodes_num), dtype=torch.bool).to(device)
        maskFalse = torch.zeros((samples, batch_size, 1), dtype=torch.bool).to(device)
        v0 = u = torch.zeros((samples, batch_size, 1), dtype=torch.long).to(device) # starts from v0:=0
        mask.scatter_(dim=-1, index=u, src=maskFalse)

        y, logp, logq, sol = [], [], [], [u]
        for i in range(1, nodes_num):
            zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((samples, batch_size, 1, nodes_num)))
            zei = zei.squeeze(dim=-2).masked_select(mask.clone()).view(samples, batch_size, nodes_num - i)
            if mode == "softmax":
                pei = F.softmax(zei, dim=-1)
                qei = epsilon / (nodes_num - i) + (1. - epsilon) * pei
                vi = qei.view(samples * batch_size, nodes_num - i)
                vi = vi.multinomial(num_samples=1, replacement=True).view(samples, batch_size, 1)
                logp.append(torch.log(pei.gather(dim=-1, index=vi) + 1e-8))
                logq.append(torch.log(qei.gather(dim=-1, index=vi) + 1e-8))
            elif mode == "greedy":
                vi = zei.argmax(dim=-1, keepdim=True)
            else:
                raise ValueError("Only support ``greedy`` and ``softmax`` mode")
            v = idx.masked_select(mask).view(samples, batch_size, nodes_num - i).gather(dim=-1, index=vi)
            y.append(adj_flat.gather(dim=-1, index = u * nodes_num + v))
            u = v
            mask.scatter_(dim=-1, index=u, src=maskFalse)
            if mode == "greedy":
                sol.append(u)
                
        y.append(adj_flat.gather(dim=-1, index=u * nodes_num + v0)) # ends at node v0
        y = torch.cat(y, dim=-1).sum(dim=-1).T # (batch_size, samples)
        
        if mode == "softmax":
            logp = torch.cat(logp, dim=-1).sum(dim=-1).T
            logq = torch.cat(logq, dim=-1).sum(dim=-1).T
            return y, logp, logq # (batch_size, samples)
        else:
            return y.squeeze(dim=1), torch.cat(sol, dim=-1).squeeze(dim=0) # (batch_size,)

    def inference_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        with torch.no_grad():
            x0_pred = self.forward(
                x=points, graph=distmat, edge_index=edge_index, timesteps=None
            )
        heatmap = x0_pred.squeeze(dim=1)
        nodes_num = heatmap.shape[-1]
        heatmap[:, range(nodes_num), range(nodes_num)] = -10  
        heatmap = heatmap.softmax(dim=-1)
        if self.env.mode == "solve":
            return heatmap
    
        # return loss if mode = `val`
        loss = self.tsp_outer_loop(points, distmat)     
        return loss, heatmap
    
    def train_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Tensor:
        return self.tsp_outer_loop(points, distmat)