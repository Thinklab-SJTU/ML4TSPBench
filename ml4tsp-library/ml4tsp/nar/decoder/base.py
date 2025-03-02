import torch
import numpy as np
import scipy.sparse
from typing import Union, Sequence
from ml4co_kit import to_numpy, check_dim, TSPSolver
from .active_search import ML4TSPNARActiveSearch


class ML4TSPNARDecoder:
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 10,
        as_samples: int = 100,
        as_inner_lr: float = 5e-2,
    ):
        self.nodes_num = None
        self.heatmap_delta = heatmap_delta
        self.active_search = active_search
        self.as_steps = as_steps
        self.as_samples = as_samples
        self.as_inner_lr = as_inner_lr
        self.device = None
        self.heatmap = None
        self.points = None
        self.edge_index = None
        self.sparse = None
    
    def check_heatmap_dim(self):
        if self.sparse:
            if self.heatmap.ndim == 1:
                self.heatmap = np.expand_dims(self.heatmap, axis=0)
            check_dim(self.heatmap, 2)
        else:
            if self.heatmap.ndim == 2:
                self.heatmap = np.expand_dims(self.heatmap, axis=0)
            check_dim(self.heatmap, 3)
        
    def check_points_dim(self):
        if self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        check_dim(self.points, 3)
        
    def check_edge_index_dim(self):
        if self.edge_index.ndim == 2:
            self.edge_index = np.expand_dims(self.edge_index, axis=0)
        check_dim(self.edge_index, 3)
    
    def sparse_to_dense(self, per_heatmap_num):
        heatmap = list()
        for idx in range(self.heatmap.shape[0]):
            _edge_index = self.edge_index[idx // per_heatmap_num]
            _max_frm_value = np.max(_edge_index[0])
            _max_to_value = np.max(_edge_index[1])
            _max_index_value = max(_max_frm_value, _max_to_value)
            _heatmap_0 = scipy.sparse.coo_matrix(
                arg1=(self.heatmap[idx], (_edge_index[0], _edge_index[1])),
                shape=(_max_index_value+1, _max_index_value+1)
            ).toarray()
            _heatmap_1 = scipy.sparse.coo_matrix(
                arg1=(self.heatmap[idx], (_edge_index[1], _edge_index[0])),
                shape=(_max_index_value+1, _max_index_value+1)
            ).toarray()
            _heatmap = (_heatmap_0 + _heatmap_1) / 2
            _heatmap = np.clip(
                a=_heatmap, 
                a_min=self.heatmap_delta,
                a_max=1-self.heatmap_delta
            )
            heatmap.append(_heatmap)
        self.heatmap = np.array(heatmap)
    
    def is_valid_tour(self, tour: np.ndarray):
        return sorted(tour[:-1]) == [i for i in range(self.nodes_num)]
    
    def heatmap_as(self):
        tsp_as = ML4TSPNARActiveSearch(
            points=self.points, num_steps=self.as_steps, num_samples=self.as_samples, 
            inner_lr=self.as_inner_lr, device=self.device
        )
        heatmap = tsp_as.active_search(self.heatmap)
        heatmap = torch.exp(heatmap)
        heatmap_list = list()
        for idx in range(heatmap.shape[0]):
            max_value = heatmap[idx].max()
            min_value = heatmap[idx].min()
            _heatmap = (heatmap[idx] - min_value) / (max_value - min_value)
            _heatmap *= (1 - 2 * self.heatmap_delta)
            _heatmap += self.heatmap_delta
            heatmap_list.append(_heatmap)
        heatmap = torch.cat(heatmap_list, dim=0)
        self.heatmap = to_numpy(heatmap.detach())
        self.heatmap = self.heatmap.reshape(-1, self.nodes_num, self.nodes_num)
        
    def decode(
        self,
        heatmap: Union[np.ndarray, torch.Tensor],
        points: Union[np.ndarray, torch.Tensor],
        edge_index: Union[np.ndarray, torch.Tensor] = None,
        per_heatmap_num: int = 1,
        return_costs: bool = False
    ) -> Union[Sequence[np.ndarray], np.floating]:
        # np.narray type
        self.heatmap = to_numpy(heatmap)
        self.points = to_numpy(points)
        self.edge_index = to_numpy(edge_index)
        self.nodes_num = self.points.shape[-2]

        # check dim of the array
        self.check_points_dim()
        
        # heatmap may be None
        if self.heatmap is not None:
            self.check_heatmap_dim()
            # sparse to dense
            if self.sparse:
                self.check_edge_index_dim()
                self.sparse_to_dense(per_heatmap_num)
            else:
                heatmap_T = self.heatmap.transpose(0, 2, 1)
                self.heatmap = (self.heatmap + heatmap_T) / 2
                self.heatmap = np.clip(
                    a=self.heatmap,
                    a_min=self.heatmap_delta,
                    a_max=1-self.heatmap_delta
                )
            # active search
            if self.active_search:
                self.heatmap_as()
        
        # decoding
        
        if per_heatmap_num > 1:
            tours_list = list()
            for idx in range(per_heatmap_num):
                tours = self._decode(
                    heatmap=self.heatmap, points=self.points[idx // per_heatmap_num]
                )
                tours_list.append(tours)
            tours = np.array(tours_list)
        else:
            tours = self._decode(heatmap=self.heatmap, points=self.points)
        batch_size, per_tours_num, tour_length = tours.shape
        tours = tours.reshape(-1, per_tours_num, tour_length)
        
        # check the tours
        for _tours in tours:
            for tour in _tours:
                if not self.is_valid_tour(tour):
                    raise ValueError(f"The tour {tour} is not valid!")
                if tour[-1] != 0:
                    raise ValueError(f"The tour {tour} is not valid!")
                
        if not return_costs:
            return tours, self.heatmap, per_tours_num
        else:
            tmp_tsp_solver = TSPSolver()
            tmp_tsp_solver.from_data(points=self.points, tours=tours[0])
            return tmp_tsp_solver.evaluate()
        
    def _decode(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        heatmap (np.ndarray): (batch_size, nodes_num, nodes_num,)
        points (np.ndarray): (batch_size, nodes_num, 2)
        """
        raise NotImplementedError("_decode is required to implemented in subclasses.")