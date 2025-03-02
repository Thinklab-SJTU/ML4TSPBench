import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm
from typing import Any, Union

from models.utils.evaluator import TSPEvaluator
from models.utils.active_search import ActiveSearch
from utils.utils import coords_to_distance
from search import tsp_greedy, get_decoding_func, get_local_search_func
from .gnn_base import MetaGNN
import pygmtools as pygm


tsp_dimes_path = {
    50:  'https://huggingface.co/ML4TSPBench/DIMES/resolve/main/tsp50_dimes.pt?download=true',
    100: 'https://huggingface.co/ML4TSPBench/DIMES/resolve/main/tsp100_dimes.pt?download=true',
    500: 'https://huggingface.co/ML4TSPBench/DIMES/resolve/main/tsp500_dimes.pt?download=true',
}


class TSPDIMES(MetaGNN):
    def __init__(
        self,
        num_nodes: int=50,
        # network
        network_type: str="gnn",
        input_dim: int=2,
        embedding_dim: int=128,
        hidden_dim: int=256,
        output_channels: int=1,
        num_layers: int=12,
        sparse_factor: int=-1, 
        # train/valid/test
        train_batch_size: int=64,
        valid_batch_size: int=1,
        test_batch_size: int=1,
        train_file: str=None,
        valid_file: str=None,
        test_file: str=None,
        valid_samples: int=1280,
        mode: str=None,
        num_workers: int=0,
        # parallel/active
        parallel_sampling: int=1,
        active_search: bool=False,
        as_steps: int=100,
        as_samples: int=2000,
        inner_epochs: int=100,
        inner_samples: int=2000,
        inner_lr: float=5e-2,
        # test_step
        decoding_type: str="greedy",
        local_search_type: str="mcts",
        **kwargs
    ):
        super(TSPDIMES, self).__init__(
            num_nodes=num_nodes,
            network_type=network_type,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_layers=num_layers,
            sparse_factor=sparse_factor,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            valid_samples=valid_samples,
            mode=mode,
            num_workers=num_workers,
            **kwargs
        )
        
        assert self.sparse == False, 'Dimes doesn\'t support sparse.'
        self.parallel_sampling = parallel_sampling
        self.active_search = active_search
        self.as_steps = as_steps
        self.as_samples = as_samples
        self.inner_lr = inner_lr
        self.inner_epochs = inner_epochs
        self.inner_samples = inner_samples

        self.test_decoding_type = decoding_type
        self.test_decoding_kwargs = kwargs
        self.test_ls_type = local_search_type
        self.test_ls_kwargs = kwargs

        self.avg_gap = 0
        self.tested = 0

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        edge_index = None
        np_edge_index = None
        if not self.sparse:
            _, points, _, gt_tour = batch
            points: torch.Tensor
            gt_tour: torch.Tensor
            # deal with different mode
            graph = coords_to_distance(points)
            if phase != "train":
                np_points = points.cpu().numpy()[0]
                np_gt_tour = gt_tour.cpu().numpy()[0]
        
        x0_pred = self.forward(points, graph, edge_index)
        if phase != 'test':
            loss = self.tsp_outer_loop(points, graph, mode=phase)
        
        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return None # customized backward already done within outer_loop
        
        # Gain the heatmap
        adj_mat = x0_pred.squeeze(dim=1)
        adj_mat: torch.Tensor

        # Decoding / solve
        if phase == "val":
            adj_mat = adj_mat.cpu().numpy()
            solved_tours = tsp_greedy(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                device=gt_tour.device,
            )
        else:
            # Active search
            if self.active_search:
                AS = ActiveSearch(graph, self.as_steps, self.as_samples, self.inner_lr)
                adj_mat = AS.active_search(adj_mat.clone().detach())
            adj_mat = adj_mat.cpu().numpy()

            # decode
            decoding_func = get_decoding_func(task="tsp", name=self.test_decoding_type)
            solved_tours = decoding_func(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                device=gt_tour.device,
                **self.test_decoding_kwargs
            )

            # local_search
            local_search_func = get_local_search_func(task="tsp", name=self.test_ls_type)
            if local_search_func is not None:
                solved_tours = local_search_func(
                    np_points=np_points, 
                    tours=solved_tours, 
                    adj_mat=adj_mat, 
                    device=gt_tour.device,
                    **self.test_ls_kwargs
                )

        # Check the tours
        for idx in range(len(solved_tours)):
            assert sorted(solved_tours[idx][:-1]) == [i for i in range(self.num_nodes)], solved_tours[idx]

        # Caculate the gap
        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(self.parallel_sampling)]
        best_solved_cost = np.min(all_solved_costs)
        gap = (best_solved_cost - gt_cost) / gt_cost * 100
        self.gap_list.append(gap)
        
        self.avg_gap = (self.avg_gap * self.tested + gap) / (self.tested + 1)
        self.tested += 1
        print(f"cur_gap: {gap:.5f}%, avg_gap: {self.avg_gap:.5f}%", end='\r' if self.tested % 20 else '\n')

        # Log the loss and gap
        metrics = {
            f"{phase}/gap": gap
        }
        if phase == 'test':
            metrics.update({
                "test/gt_cost": gt_cost,
                "test/solved_cost": best_solved_cost
            })
        else:
            metrics.update({
                f"{phase}/loss": loss
            })
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def solve(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        edge_index: Union[np.ndarray, torch.Tensor]=None,
        batch_size: int=16, 
        device='cpu'
    ):
        """solve function, return heatmap"""
        self.model.eval()
        if type(data) == np.ndarray:
            data = torch.Tensor(data)
        assert self.sparse == False, 'Dimes doesn\'t support sparse.'
        
        assert data.shape[0] % batch_size == 0, f'batch_size must be divided by the size of the data'
        batch_data = data.reshape(-1, batch_size, data.shape[-2], 2)
        batch_heatmap = list()
        
        with torch.no_grad():
            for points in tqdm(batch_data, total=batch_data.shape[0], desc='Processing'):
                points = points.to(device=device)
                graph = coords_to_distance(points).to(device=device)
                x0_pred = self.forward(points, graph, edge_index=None)
                adj_mat = x0_pred.squeeze(dim=1)
                if self.active_search:
                    AS = ActiveSearch(graph, 
                                    self.as_steps, 
                                    self.as_samples, 
                                    inner_lr=self.inner_lr)
                    adj_mat = AS.active_search(emb0=adj_mat)
                adj_mat = adj_mat.detach().cpu().numpy()
                batch_heatmap.append(adj_mat)

            batch_heatmap = np.array(batch_heatmap)
            heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-2], batch_heatmap.shape[-1])
        
        return heatmap

    def load_ckpt(self, ckpt_path: str=None):
        """load state dict from checkpoint"""
        if ckpt_path is None:
            if self.num_nodes in [50, 100, 500]:
                url = tsp_dimes_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_dimes.pt"
                pygm.utils.download(filename=filename, url=url, to_cache=None)
                self.load_state_dict(torch.load(filename))
            else:
                raise ValueError(f"There is currently no pretrained checkpoint with {self.num_nodes} nodes.")
        else:
            checkpoint = torch.load(ckpt_path)
            if ckpt_path.endswith('.ckpt'):
                state_dict = checkpoint['state_dict']
            elif ckpt_path.endswith('.pt'):
                state_dict = checkpoint
            self.load_state_dict(state_dict)
            
    def tsp_sample(self, dist_mat, ze, mode='softmax', samples=1, epsilon=0.): # epsilon exploration
        assert mode in ['softmax', 'greedy']
        if mode == 'greedy':
            assert samples == 1
        batch_size, n_nodes, _ = dist_mat.shape
        zex = ze.expand((samples, batch_size, n_nodes, n_nodes))
        adj_flat = dist_mat.view(batch_size, n_nodes * n_nodes).expand((samples, batch_size, n_nodes * n_nodes))
        idx = torch.arange(n_nodes).expand((samples, batch_size, n_nodes)).to(self.device)
        mask = torch.ones((samples, batch_size, n_nodes), dtype=torch.bool).to(self.device)
        maskFalse = torch.zeros((samples, batch_size, 1), dtype=torch.bool).to(self.device)
        v0 = u = torch.zeros((samples, batch_size, 1), dtype=torch.long).to(self.device) # starts from v0:=0
        mask.scatter_(dim=-1, index=u, src=maskFalse)
        y = []
        if mode == 'softmax':
            logp, logq = [], []
        else:
            sol = [u]
        for i in range(1, n_nodes):
            zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((samples, batch_size, 1, n_nodes))).squeeze(dim=-2).masked_select(mask.clone()).view(samples, batch_size, n_nodes - i)
            if mode == 'softmax':
                pei = F.softmax(zei, dim=-1)
                qei = epsilon / (n_nodes - i) + (1. - epsilon) * pei
                vi = qei.view(samples * batch_size, n_nodes - i).multinomial(num_samples=1, replacement=True).view(samples, batch_size, 1)
                logp.append(torch.log(pei.gather(dim=-1, index=vi) + 1e-8))
                logq.append(torch.log(qei.gather(dim=-1, index=vi) + 1e-8))
            elif mode == 'greedy':
                vi = zei.argmax(dim=-1, keepdim=True)
            v = idx.masked_select(mask).view(samples, batch_size, n_nodes - i).gather(dim=-1, index = vi)
            y.append(adj_flat.gather(dim=-1, index = u * n_nodes + v))
            u = v
            mask.scatter_(dim=-1, index=u, src=maskFalse)
            if mode == 'greedy':
                sol.append(u)
        y.append(adj_flat.gather(dim=-1, index=u * n_nodes + v0)) # ends at node v0
        y = torch.cat(y, dim=-1).sum(dim=-1).T # (batch_size, samples)
        if mode == 'softmax':
            logp = torch.cat(logp, dim=-1).sum(dim=-1).T
            logq = torch.cat(logq, dim=-1).sum(dim=-1).T
            return y, logp, logq # (batch_size, samples)
        elif mode == 'greedy':
            return y.squeeze(dim=1), torch.cat(sol, dim=-1).squeeze(dim=0) # (batch_size,)

    def tsp_greedy(self, dist_mat, ze):
        return self.tsp_sample(dist_mat, ze, mode='greedy') # y, sol

    def tsp_inner_loop(self, dist_mat, emb0, steps, samples, epsilon=0., points=None):
        batch_size, _, _ = dist_mat.shape
        emb = nn.Parameter(emb0.detach().clone(), requires_grad=True)
        inner_opt = torch.optim.AdamW([emb], lr=self.inner_lr, weight_decay=0.)
        y_means = []
        y_bl = torch.zeros((batch_size, 1))
        for t in range(1, steps + 1): #tbar:
            inner_opt.zero_grad()
            heatmap = emb
            y, logp, logq = self.tsp_sample(dist_mat, heatmap, 'softmax', samples, epsilon)
            y_means.append(y.mean().item())
            y_bl = y.mean(dim=-1, keepdim=True)
            J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim=-1).sum()
            J.requires_grad_(True)
            J.backward()
            inner_opt.step()

        return emb, y_means
    
    def tsp_outer_loop(self, points, dist_mat, mode='train'):
        if mode == 'train':
            batch_size, _, _ = dist_mat.shape
            opt = self.optimizers()
            opt.zero_grad()
            par0 = self.forward(points, dist_mat, edge_index=None)
            par0 = par0.squeeze(dim=1)
            loss = self.tsp_greedy(dist_mat, par0)[0].mean().item()
            par1, _ = self.tsp_inner_loop(dist_mat, par0, self.inner_epochs, self.inner_samples)
            par0.backward(par1.grad / batch_size)
            opt.step()

        else: # for validation or test
            par0 = self.forward(points, dist_mat, edge_index=None)
            par0 = par0.squeeze(dim=1)
            loss = self.tsp_greedy(dist_mat, par0)[0].mean().item()

        return loss
