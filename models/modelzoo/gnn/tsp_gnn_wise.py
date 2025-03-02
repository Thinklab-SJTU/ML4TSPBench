import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from tqdm import tqdm
from typing import Any, Union
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data as GraphData
from torch_sparse import SparseTensor

from models.utils.evaluator import TSPEvaluator
from models.utils.active_search import ActiveSearch
from utils.utils import coords_to_distance
from search import tsp_greedy, get_decoding_func, get_local_search_func
from .gnn_base import MetaGNN
import pygmtools as pygm


tsp_gnn_wise_path = {
    50:  'https://huggingface.co/Bench4CO/GNN-WISE/resolve/main/TSP/tsp50_gnn_wise.ckpt?download=true',
    100: 'https://huggingface.co/Bench4CO/GNN-WISE/resolve/main/TSP/tsp100_gnn_wise.ckpt?download=true',
    500: 'https://huggingface.co/Bench4CO/GNN-WISE/resolve/main/TSP/tsp500_gnn_wise.ckpt?download=true',
}


class TSPGNNWISE(MetaGNN):
    def __init__(
        self,
        num_nodes: int=50,
        # network
        network_type: str="gnn",
        input_dim: int=2,
        embedding_dim: int=128,
        hidden_dim: int=256,
        output_channels: int=2,
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
        mode: str="train",
        num_workers: int=0,
        # parallel/active
        parallel_sampling: int=1,
        active_search: bool=False,
        as_steps: int=100,
        as_samples: int=1000,
        inner_lr: float=5e-2,
        # precision
        fp16: bool=False,
        # test_step
        decoding_type: str="greedy",
        local_search_type: str=None,
        **kwargs
    ):
        super(TSPGNNWISE, self).__init__(
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

        self.parallel_sampling = parallel_sampling
        self.active_search = active_search
        self.as_steps = as_steps
        self.as_samples = as_samples
        self.inner_lr = inner_lr
        self.fp16 = fp16
        self.wise_epsilon = 1e-6 if self.fp16 else 1e-14

        self.decoding_type = decoding_type
        self.test_decoding_kwargs= kwargs
        self.local_search_type = local_search_type
        self.test_ls_kwargs = kwargs
        
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        edge_index = None
        np_edge_index = None
        if not self.sparse:
            _, points, adj_matrix, gt_tour = batch
            points: torch.Tensor
            gt_tour: torch.Tensor
            graph = coords_to_distance(points)
            # deal with different mode
            if phase != "train":
                np_points = points.cpu().numpy()[0]
                np_gt_tour = gt_tour.cpu().numpy()[0]
        else:
            graph_data = batch[1]
            graph_data: GraphData
            gt_tour = batch[4]
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            edge_index: torch.Tensor
            adj_matrix = route_edge_flags.reshape((1, -1))[0]
            graph = coords_to_distance(points, edge_index)
            if phase != "train":
                np_points = points.cpu().numpy()
                np_gt_tour = gt_tour.reshape(-1).cpu().numpy()
                np_edge_index = edge_index.cpu().numpy()

        x0_pred = self.forward(points, graph, edge_index)

        # Caculate loss
        x0_pred: torch.Tensor
        adj_matrix: torch.Tensor
        edge_labels = adj_matrix.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edge_cw = torch.Tensor(edge_cw).to(x0_pred.device)

        if not self.sparse: 
            x0 = F.softmax(x0_pred, dim=-1)
            x0_0 = (x0[:, 0, :, :] * (self.num_nodes - 2)).unsqueeze(dim=1)
            x0_1 = (x0[:, 1, :, :] * 2).unsqueeze(dim=1)
        else:
            x0_0_reshape = x0_pred[:, 0].reshape(-1, self.sparse_factor)
            x0_0_softmax = F.softmax(x0_0_reshape, dim=-1) * (self.sparse_factor - 2)
            x0_0 = x0_0_softmax.reshape(-1, 1)
            x0_1_reshape = x0_pred[:, 1].reshape(-1, self.sparse_factor)
            x0_1_softmax = F.softmax(x0_1_reshape, dim=-1) * 2
            x0_1 = x0_1_softmax.reshape(-1, 1)

        h0 = torch.cat([x0_0, x0_1], dim=1)
        heatmap = torch.clamp(h0, self.wise_epsilon, 1)
        log_heatmap = torch.log(heatmap)
        loss = nn.NLLLoss(edge_cw)(log_heatmap, adj_matrix.long())
        
        # return loss if current is a training step
        if phase == "train":
            metrics = {"train/loss": loss}
            for k, v in metrics.items():
                self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss
        
        # Gain the heatmap
        if not self.sparse:
            adj_mat = heatmap[:, 1, :, :]
        else:
            adj_mat = heatmap[:, 1]    

        # Decoding / solve
        if phase == "val":
            adj_mat = adj_mat.cpu().numpy()
            solved_tours = tsp_greedy(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                device=gt_tour.device
            )
        else:
            # Active search
            if self.active_search:
                dist_mat = torch.cdist(points, points)
                AS = ActiveSearch(dist_mat, self.as_steps, self.as_samples, self.inner_lr)
                adj_mat = AS.active_search(adj_mat.clone().detach())
                adj_mat = torch.clamp(adj_mat, min=1e-14)
            adj_mat = adj_mat.cpu().numpy()
            
            # decode
            decoding_func = get_decoding_func(task="tsp", name=self.decoding_type)
            solved_tours = decoding_func(
                adj_mat=adj_mat, 
                np_points=np_points, 
                edge_index_np=np_edge_index, 
                sparse_graph=self.sparse, 
                device=gt_tour.device,
                **self.test_decoding_kwargs
            )

            # local_search
            local_search_func = get_local_search_func(task="tsp", name=self.local_search_type)
            if local_search_func is not None:
                if self.sparse:
                    if self.parallel_sampling == 1:
                        sparse_adj_mat = SparseTensor(
                            row=edge_index[0],
                            col=edge_index[1],
                            value=torch.tensor(adj_mat).to(device=edge_index.device)
                        )
                        adj_mat = sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()
                    else:
                        sparse_adj_mat = list()
                        ps_edge_index = edge_index.reshape(2, self.parallel_sampling, -1).transpose(0, 1)
                        ps_adj_mat = adj_mat.reshape(self.parallel_sampling, -1)
                        for idx in range(self.parallel_sampling):
                            ps_sparse_adj_mat = SparseTensor(
                                row=ps_edge_index[idx][0] - self.num_nodes*idx,
                                col=ps_edge_index[idx][1] - self.num_nodes*idx,
                                value=torch.tensor(ps_adj_mat[idx]).to(device=edge_index.device)
                            )
                            sparse_adj_mat.append(ps_sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()[0])
                        adj_mat = np.array(sparse_adj_mat)

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
        
        if not self.sparse:
            batch_data = data.reshape(-1, batch_size, data.shape[-2], 2)
            batch_edge_index = [None] * batch_data.shape[0]
        else:
            batch_num = data.shape[0] // batch_size
            batch_data = data.reshape(batch_num, -1, 2)
            batch_edge_index = edge_index.reshape(batch_num, -1, 2, edge_index.shape[-1])
            batch_edge_index = batch_edge_index.transpose(1, 2)
            batch_edge_index = batch_edge_index.reshape(batch_num, 2, -1)
            length = self.sparse_factor * self.num_nodes
            for idx in range(batch_size):
                batch_edge_index[:, :, idx*length:(idx+1)*length] += (idx * self.num_nodes) 
        
        batch_heatmap = list()
        with torch.no_grad():
            for idx, points in tqdm(enumerate(batch_data), total=batch_data.shape[0], desc='Processing'):
                points = points.to(device=device)
                graph = coords_to_distance(points, batch_edge_index[idx]).to(device=device)
                x0_pred = self.forward(points, graph, batch_edge_index[idx])
                if not self.sparse: 
                    x0 = F.softmax(x0_pred, dim=-1)
                    x0_0 = (x0[:, 0, :, :] * (self.num_nodes - 2)).unsqueeze(dim=1)
                    x0_1 = (x0[:, 1, :, :] * 2).unsqueeze(dim=1)
                else:
                    x0_0_reshape = x0_pred[:, 0].reshape(-1, self.sparse_factor)
                    x0_0_softmax = F.softmax(x0_0_reshape, dim=-1) * (self.sparse_factor - 2)
                    x0_0 = x0_0_softmax.reshape(-1, 1)
                    x0_1_reshape = x0_pred[:, 1].reshape(-1, self.sparse_factor)
                    x0_1_softmax = F.softmax(x0_1_reshape, dim=-1) * 2
                    x0_1 = x0_1_softmax.reshape(-1, 1)
                h0 = torch.cat([x0_0, x0_1], dim=1)
                heatmap = torch.clamp(h0, self.wise_epsilon, 1)
                if not self.sparse:
                    heatmap = heatmap[:, 1, :, :]
                else:
                    heatmap = heatmap[:, 1].reshape(batch_size, -1)  
                heatmap = heatmap.cpu().detach().numpy()
                batch_heatmap.append(heatmap)

        batch_heatmap = np.array(batch_heatmap)
        if not self.sparse:
            heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-2], batch_heatmap.shape[-1])
        else:
            heatmap = batch_heatmap.reshape(-1, batch_heatmap.shape[-1])
        return heatmap

    def load_ckpt(self, ckpt_path:str=None):
        if ckpt_path is None:
            if self.num_nodes in [50, 100, 500]:
                url = tsp_gnn_wise_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_gnn_wise.ckpt"
                pygm.utils.download(filename=filename, url=url, to_cache=None)
                self.load_state_dict(torch.load(filename)['state_dict'])
            else:
                raise ValueError(f"There is currently no pretrained checkpoint with {self.num_nodes} nodes.")
        else:
            checkpoint = torch.load(ckpt_path)
            if ckpt_path.endswith('.ckpt'):
                state_dict = checkpoint['state_dict']
            elif ckpt_path.endswith('.pt'):
                state_dict = checkpoint
            self.load_state_dict(state_dict)