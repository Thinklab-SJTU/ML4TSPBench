import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm
from typing import Any, Union
from torch_geometric.data import Data as GraphData
from torch_sparse import SparseTensor

from models.utils.evaluator import TSPEvaluator
from utils.utils import coords_to_distance
from search import tsp_greedy, get_decoding_func, get_local_search_func
from .gnn_base import MetaGNN
import pygmtools as pygm
import os


tsp_gnn_path = {
    50:  'https://huggingface.co/Bench4CO/gnn4reg/resolve/main/TSP/tsp50_gnn4reg.ckpt?download=true',
    100: 'https://huggingface.co/Bench4CO/gnn4reg/resolve/main/TSP/tsp100_gnn4reg.ckpt?download=true',
}


class TSPGNN4REG(MetaGNN):
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
        regret_dir: str=None,
        valid_samples: int=1280,
        mode: str=None,
        num_workers: int=0,
        parallel_sampling: int=1,
        # test_step
        decoding_type: str="greedy",
        local_search_type: str="gls",
        **kwargs
    ):
        super(TSPGNN4REG, self).__init__(
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
            regret_dir=regret_dir,
            valid_samples=valid_samples,
            mode=mode,
            num_workers=num_workers,
            **kwargs
        )

        self.parallel_sampling = parallel_sampling

        self.test_decoding_type = decoding_type
        self.test_decoding_kwargs= kwargs
        self.test_ls_type = local_search_type
        self.test_ls_kwargs = kwargs

        # self.avg_gap = 0
        # self.tested = 0

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        edge_index = None
        np_edge_index = None
        if phase == 'test':
            _, points, _, gt_tour = batch
        else:
            gt_reg_mat, _, points, _, gt_tour = batch
        points: torch.Tensor
        gt_reg_mat: torch.Tensor
        # deal with different mode
        graph = coords_to_distance(points)
        # graph = torch.cdist(points, points)
        if phase == 'test':
            points = points.unsqueeze(dim=0)
            graph = graph.unsqueeze(dim=0)
        np_points = points.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()
        x0_pred = self.forward(points, graph, edge_index)
        reg_mat = torch.triu(x0_pred.squeeze(1), diagonal=1)

        # Caculate loss
        if phase in ['train', 'val']:
            reg_mat: torch.Tensor
            loss_func = nn.MSELoss()
            loss = loss_func(reg_mat, gt_reg_mat.triu(diagonal=1))
        
        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss 
        
        ########### train_step ends here ##########
        
        # Decoding / solve
        reg_mat = x0_pred.squeeze()
        edge_reg = torch.masked_select(reg_mat, self.test_dataset.mask.to(reg_mat.device)).view(-1, 1) # shape: [# edges, 1]
        reg_pred = self.test_dataset.reg_scaler.inverse_transform(edge_reg.cpu().numpy())
        reg_mat[self.test_dataset.mask] = torch.from_numpy(reg_pred.reshape(-1)).to(reg_mat.device)
        reg_mat = torch.triu(reg_mat, diagonal=1) + torch.tril(reg_mat.t(), diagonal=-1)
        reg_mat = -1 * reg_mat # for decoding consistency, and will be recovered if using gls as local search method
        reg_mat = reg_mat.unsqueeze(dim=0).cpu().numpy()

        if phase == "val":
            reg_mat = reg_mat.cpu().numpy()
            solved_tours = tsp_greedy(
                adj_mat=reg_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                device=gt_tour.device,
            )
        else:  # phase == "test"
            # decode
            decoding_func = get_decoding_func(task="tsp", name=self.test_decoding_type)
            solved_tours = decoding_func(
                adj_mat=reg_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                device=gt_tour.device,
                **self.test_decoding_kwargs
            )

            # local_search
            local_search_func = get_local_search_func(task="tsp", name=self.test_ls_type)
            if local_search_func is not None:
                solved_tours = local_search_func(
                    np_points=np_points, 
                    tours=solved_tours, 
                    adj_mat=reg_mat, 
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
        self.gap_list.append(gap)
        
        # self.avg_gap = (self.avg_gap * self.tested + gap) / (self.tested + 1)
        # self.tested += 1
        # print(f"cur_gap: {gap:.5f}%, avg_gap: {self.avg_gap:.5f}%", end='\r' if self.tested % 20 else '\n')

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

    def solve(self, data: Union[np.ndarray, torch.Tensor], batch_size: int=16, device='cpu'):
        """solve function, return heatmap"""
        import pickle
        self.model.eval()
        if type(data) == np.ndarray: # coords of points
            data = torch.Tensor(data)
        batch_data = data.reshape(-1, batch_size, data.shape[-2], data.shape[-1])
        batch_heatmap = list()
        mask = torch.triu(torch.ones(batch_size, self.num_nodes, self.num_nodes), diagonal=1).bool()
        reg_scaler = pickle.load(open(os.path.join(self.regret_dir, f'tsp{self.num_nodes}_regret_scaler.pkl'), 'rb'))
        for data in tqdm(batch_data, total=batch_data.shape[0], desc='Processing'):
            data = data.to(device=device)
            graph = coords_to_distance(data).to(device=device)
            x0_pred = self.forward(data, graph, edge_index=None)
            
            reg_mat = x0_pred.squeeze()
            edge_reg = torch.masked_select(reg_mat, mask.to(reg_mat.device)).view(-1, 1) # shape: [# edges, 1]
            reg_pred = reg_scaler.inverse_transform(edge_reg.cpu().detach().numpy())
            reg_mat[mask] = torch.from_numpy(reg_pred.reshape(-1)).to(reg_mat.device)
            reg_mat = torch.triu(reg_mat, diagonal=1) + torch.tril(reg_mat.transpose(1, 2), diagonal=-1)
            reg_mat = -1 * reg_mat # for decoding consistency, and will be recovered if using gls as local search method

            reg_mat = reg_mat.cpu().detach().numpy()
            batch_heatmap.append(reg_mat)

        batch_heatmap = np.array(batch_heatmap)
        heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-2], batch_heatmap.shape[-1])
        
        return heatmap

    def load_ckpt(self, ckpt_path: str=None):
        """load state dict from checkpoint"""
        if ckpt_path is None:
            if self.num_nodes in [50, 100]:
                url = tsp_gnn_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_gnn4reg.pt"
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