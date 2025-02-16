import os
import torch
import pickle
import numpy as np
from torch import Tensor
from typing import Sequence
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from ml4co_kit import BaseEnv, to_numpy, to_tensor, check_dim, points_to_distmat
from ml4tsp.nar.env.dataset import ML4TSPGraphDataset


class ML4TSPNAREnv(BaseEnv):
    def __init__(
        self,
        nodes_num: int = None,
        mode: str = None,
        train_path: str = None,
        val_path: str = None,
        train_batch_size: int = 4,
        regret_path: str = None,
        num_workers: int = 4,
        sparse_factor: int = -1,
        device: str = "cpu"
    ):
        super().__init__(
            name="ML4TSPNAREnv",
            mode=mode,
            train_path=train_path,
            val_path=val_path,
            train_batch_size=train_batch_size,
            num_workers=num_workers,
            device=device
        )
        self.nodes_num = nodes_num
        
        # sparse
        self.sparse_factor = sparse_factor
        self.sparse = self.sparse_factor > 0
        
        # regret
        self.reg_scaler = None
        self.regret_path = regret_path
        self.regret_mask = None
        if self.regret_path is not None:
            pkl_path = f"ml4tsp/nar/env/gnn_reg_scaler/tsp{self.nodes_num}_regret_scaler.pkl"
            self.reg_scaler = pickle.load(open(pkl_path, 'rb'))
            self.regret_mask = torch.triu(torch.ones(self.nodes_num, self.nodes_num), diagonal=1).bool()
        
        # load data
        self.load_data()
        
    def load_data(self):
        if self.mode == "train":
            self.train_dataset = ML4TSPGraphDataset(
                file_path=self.train_path,
                reg_path=self.regret_path,
                reg_scaler=self.reg_scaler,
                reg_mask=self.regret_mask,
                mode=self.mode
            )
            self.val_dataset = ML4TSPGraphDataset(
                file_path=self.val_path,
                reg_path=self.regret_path,
                reg_scaler=self.reg_scaler,
                reg_mask=self.regret_mask,
                mode=self.mode
            )
            
    def train_dataloader(self):
        train_dataloader=DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader

    def _sparse_process_data(self, points: Tensor, ref_tour: Tensor) -> Sequence[Tensor]:
        # check dim
        check_dim(points, 2)
        check_dim(ref_tour, 1)
        
        # to numpy
        points = to_numpy(points)
        ref_tour = to_numpy(ref_tour)
        
        # KDTree        
        kdt = KDTree(points, leaf_size=30, metric='euclidean')
        idx_knn = kdt.query(points, k=self.sparse_factor, return_distance=False)

        # edge_index
        edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1))
        edge_index_0 = edge_index_0.repeat(1, self.sparse_factor).reshape(-1)
        edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        # ground_truth
        tour_edges = np.zeros(points.shape[0], dtype=np.int64)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, self.sparse_factor).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        
        tour_edges_rv = np.zeros(points.shape[0], dtype=np.int64)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv = torch.from_numpy(tour_edges_rv)
        tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, self.sparse_factor).reshape(-1)
        tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)

        ground_truth = tour_edges + tour_edges_rv
        
        # to tensor
        edge_index = to_tensor(edge_index)
        ground_truth = to_tensor(ground_truth)
              
        return ground_truth, edge_index
    
    def sparse_process_data(self, points: Tensor, ref_tours: Tensor) -> Sequence[Tensor]:
        # check dim
        check_dim(points, 3)
        check_dim(ref_tours, 2)
        batch_size = points.shape[0]
        
        # process data
        ground_truth_list = list()
        edge_index_list = list()  
        for idx in range(points.shape[0]):
            ground_truth, edge_index = self._sparse_process_data(points[idx], ref_tours[idx])
            ground_truth_list.append(ground_truth)
            edge_index_list.append(edge_index)
        
        # torch.cat 
        edge_index = torch.cat(edge_index_list, dim=0).to(self.device)
        edge_index = edge_index.reshape(batch_size, 2, -1)
        ground_truth = torch.cat(ground_truth_list, dim=0).to(self.device)
        ground_truth = ground_truth.reshape(batch_size, -1)
        distmat = points_to_distmat(points, edge_index)
        
        # return
        return points, edge_index, distmat, ground_truth

    def _dense_process_data(self, points: Tensor, ref_tour: Tensor) -> Sequence[Tensor]:
        # check dim
        check_dim(points, 2)
        check_dim(ref_tour, 1)
        
        # to numpy
        points = to_numpy(points)
        ref_tour = to_numpy(ref_tour)
        
        # adj matrix
        nodes_num = points.shape[0]
        ground_truth = np.zeros((nodes_num, nodes_num))
        for i in range(ref_tour.shape[0] - 1):
            ground_truth[ref_tour[i], ref_tour[i + 1]] = 1
            ground_truth[ref_tour[i + 1], ref_tour[i]] = 1
        
        # to tensor
        ground_truth = to_tensor(ground_truth)
        return ground_truth

    def dense_process_data(self, points: Tensor, ref_tours: Tensor, ref_regret: Tensor) -> Sequence[Tensor]:
        # check dim
        check_dim(points, 3)
        check_dim(ref_tours, 2)
        check_dim(ref_regret, 3)
        
        # process data
        if ref_regret is not None:
            ground_truth = ref_regret
        else:
            ground_truth_list = list() 
            for idx in range(points.shape[0]):
                ground_truth = self._dense_process_data(points[idx], ref_tours[idx])
                ground_truth_list.append(ground_truth)
            
            # torch.cat
            ground_truth = torch.cat(ground_truth_list, dim=0).to(self.device)
            ground_truth = ground_truth.reshape(-1, self.nodes_num, self.nodes_num)
        distmat = points_to_distmat(points, None)

        # return
        return points, None, distmat, ground_truth
    
    def process_data(self, points: Tensor, ref_tours: Tensor, ref_regret: Tensor=None) -> Sequence[Tensor]:
        if self.sparse:
            return self.sparse_process_data(points, ref_tours)
        else:
            return self.dense_process_data(points, ref_tours, ref_regret)