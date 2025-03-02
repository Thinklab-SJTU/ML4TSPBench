import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm
from typing import Any, Union
from torch_geometric.data import Data as GraphData
from pytorch_lightning.utilities import rank_zero_info
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt

from models.utils.evaluator import TSPEvaluator
from models.utils.active_search import ActiveSearch
from utils.utils import coords_to_distance
from search import tsp_greedy, get_decoding_func, get_local_search_func
from .gnn_base import MetaGNN
import pygmtools as pygm


utsp_path = {
    50:  'https://huggingface.co/ML4TSPBench/US/resolve/main/tsp50_us.pt?download=true',
    100: 'https://huggingface.co/ML4TSPBench/US/resolve/main/tsp100_us.pt?download=true',
    500: 'https://huggingface.co/ML4TSPBench/US/resolve/main/tsp500_us.pt?download=true'
}


class UTSPGNN(MetaGNN):
    def __init__(
        self,
        num_nodes: int=50,
        # network
        network_type: str="sag",
        input_dim: int=2,
        embedding_dim: int=128,
        hidden_dim: int=64,
        output_channels: int=1,
        num_layers: int=3,
        sparse_factor: int=-1, 
        # utsp
        temperature: float=3.5,
        distance_loss: float=1.0,
        loop_loss: float=0.1,
        row_loss: float=10.0,
        # train/valid/test
        train_batch_size: int=64,
        valid_batch_size: int=1,
        test_batch_size: int=1,
        train_file: str=None,
        valid_file: str=None,
        test_file: str=None,
        valid_samples: int=1280,
        mode: str="train",
        num_workers: int=0,
        # parallel/active
        parallel_sampling: int=1,
        active_search: bool=False,
        as_steps: int=500,
        as_samples: int=1000,
        # test_step
        decoding_type: str="greedy",
        local_search_type: str=None,
        **kwargs
    ):
        if sparse_factor >= 0:
            raise NotImplementedError("UTSP doesn't support sparse now!")
        if network_type == 'sag':
            if output_channels != num_nodes:
                rank_zero_info('Using the SAG network requires the output_channels to be the same as the num_nodes.')
                output_channels = num_nodes
        if network_type == 'gnn':
            if output_channels != 1:
                rank_zero_info('Using the GNN network requires the output_channels to be 1.')
                output_channels = 1

        super(UTSPGNN, self).__init__(
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

        self.temperature = temperature
        self.distance_loss = distance_loss
        self.loop_loss = loop_loss
        self.row_loss = row_loss

        self.parallel_sampling = parallel_sampling
        self.active_search = active_search
        self.as_steps = as_steps
        self.as_samples = as_samples

        self.test_decoding_type = decoding_type
        self.test_decoding_kwargs= kwargs
        self.test_ls_type = local_search_type
        self.test_ls_kwargs = kwargs
        
        self.mask = torch.ones(self.num_nodes, self.num_nodes)
        self.mask.fill_diagonal_(0)
        
    def utsp_loss(self, x0, dist_matrix):
        # H = TVT^T
        heatmap = torch.matmul(x0, torch.roll(torch.transpose(x0, 1, 2), -1, 1))
        # Tour-Distance loss
        distance_loss = self.distance_loss * torch.sum(torch.mul(heatmap, dist_matrix).sum(dim=(1, 2)))
        # No-Self-Loops loss
        heatmap_diagonals = torch.stack([torch.diagonal(mat) for mat in heatmap], dim=0)
        loop_loss = self.loop_loss * torch.sum(heatmap_diagonals)
        # Row-Wise-Constraint loss
        row_wise = torch.sum((1. - torch.sum(x0, 2)) ** 2)
        row_wise_loss = self.row_loss * row_wise
        # sum up
        loss = distance_loss + loop_loss + row_wise_loss
        return loss, heatmap
    
    def save_heatmap(self, heatmap: torch.Tensor, filename="heatmap.png"):
        if not os.path.exists("us_images"):
            os.mkdir("us_images")
        if heatmap.ndim == 3:
            if heatmap.shape[0] == 1:
                heatmap = heatmap[0]
            else:
                raise ValueError("heatmap must be 2D matrix!")
        heatmap = heatmap.cpu().detach().numpy()
        np.fill_diagonal(heatmap, 0)
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig("us_images/" + filename)
        plt.clf()
    
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        edge_index = None
        np_edge_index = None
        _, points, _, gt_tour = batch
        points: torch.Tensor
        gt_tour: torch.Tensor
        graph = coords_to_distance(points)
        # deal with different mode
        if phase != "train":
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
        
        graph_adj = torch.exp(-1.*graph/self.temperature)
        self.mask = self.mask.to(graph_adj.device)
        graph_adj = graph_adj * self.mask
        x0_pred = self.forward(points, graph_adj, edge_index)

        # Caculate loss
        x0_pred: torch.Tensor
        if self.network_type == 'sag':
            x0 = x0_pred
        else:
            x0 = F.softmax(x0_pred.squeeze(dim=1), dim=1)
        bacth_loss, heatmap = self.utsp_loss(x0, graph)
        loss = bacth_loss / points.shape[0]
        
        # save heatmap
        if batch_idx % 25 == 0 and phase == 'train':
            filename = "heatmap_{}".format(batch_idx)
            self.save_heatmap(heatmap[0], filename)

        # ours method to improve utsp
        heatmap = heatmap * self.mask

        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss

        # Decoding / solve
        adj_mat = heatmap
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
                dist_mat = torch.cdist(points, points)
                AS = ActiveSearch(dist_mat, self.as_steps, self.as_samples)
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
                if self.sparse:
                    sparse_adj_mat = SparseTensor(
                    row=edge_index[0],
                    col=edge_index[1],
                    value=torch.tensor(adj_mat).to(device=edge_index.device)
                    )
                    adj_mat = sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()
                solved_tours = local_search_func(
                    np_points=np_points, 
                    tours=solved_tours, 
                    adj_mat=adj_mat, 
                    device=gt_tour.device,
                    **self.test_ls_kwargs
                )

        # Check the tours
        for idx in range(len(solved_tours)):
            assert sorted(solved_tours[idx][:-1]) == [i for i in range(self.num_nodes)]

        # Caculate the gap
        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(self.parallel_sampling)]
        best_solved_cost = np.min(all_solved_costs)
        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        # record the better/worse/match
        better = 0.0
        match = 0.0
        worse = 0.0
        if gap < -1e-12:
            better = 1.0
        elif gap < 1e-12:
            match = 1.0
        else:
            worse = 1.0
        self.gap_list.append(gap)
        
        # Log the loss and gap
        metrics = {
            f"{phase}/loss": loss,
            f"{phase}/gap": gap,
            f"{phase}/better": better,
            f"{phase}/match": match,
            f"{phase}/worse": worse,
        }

        if phase == 'test':
            metrics.update({
                "test/gt_cost": gt_cost,
                "test/solved_cost": best_solved_cost
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
        if self.sparse:
            if type(edge_index) == np.ndarray:
                edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.to(device)

        assert data.shape[0] % batch_size == 0, f'batch_size must be divided by the size of the data'

        batch_data = data.reshape(-1, batch_size, data.shape[-2], 2)
        batch_heatmap = list()

        for points in tqdm(batch_data, total=batch_data.shape[0], desc='Processing'):
            points = points.to(device=device)
            graph = coords_to_distance(points, None).to(device=device)
            x0_pred = self.forward(points, graph, None)
            
            # deal with different network type
            if self.network_type == 'sag':
                x0 = x0_pred
            else:
                x0 = F.softmax(x0_pred.squeeze(dim=1), dim=1)
            _, heatmap = self.utsp_loss(x0, graph)
            heatmap = heatmap.cpu().detach().numpy()
            batch_heatmap.append(heatmap)

        batch_heatmap = np.array(batch_heatmap)
        heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-2], batch_heatmap.shape[-1])

        return heatmap

    def load_ckpt(self, ckpt_path: str=None):
        if ckpt_path is None:
            if self.num_nodes in [50, 100]:
                url = utsp_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_us.pt"
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