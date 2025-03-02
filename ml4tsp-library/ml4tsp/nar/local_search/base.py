import torch
import numpy as np
from typing import Union
from ml4co_kit import to_numpy, check_dim, iterative_execution


class ML4TSPNARLocalSearch:
    def __init__(self, time_limit: Union[float, str] = "auto"):
        self.time_limit = time_limit
        self.nodes_num = None
        self.device = None
        self.tours = None
        self.heatmap = None
        self.points = None

    def check_tours_dim(self):
        if self.tours.ndim == 1:
            self.tours = np.expand_dims(self.tours, axis=0)
        check_dim(self.tours, 2)
    
    def check_heatmap_dim(self):
        if self.heatmap.ndim == 2:
            self.heatmap = np.expand_dims(self.heatmap, axis=0)
        check_dim(self.heatmap, 3)
        
    def check_points_dim(self):
        if self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        check_dim(self.points, 3)    

    def is_valid_tour(self, tour: np.ndarray):
        return sorted(tour[:-1]) == [i for i in range(self.nodes_num)]

    def local_search( 
        self,
        tours: Union[np.ndarray, torch.Tensor],
        points: Union[np.ndarray, torch.Tensor],
        heatmap: Union[np.ndarray, torch.Tensor] = None,
        per_tours_num: int = 1,
        per_heatmap_num: int = 1,
    ) -> np.ndarray:
        # np.narray type
        self.tours = to_numpy(tours)
        self.heatmap = to_numpy(heatmap)
        self.points = to_numpy(points)
        
        # check dim of the array
        self.check_tours_dim()
        self.check_points_dim()
        if heatmap is not None:
            use_heatmap = True
            self.check_heatmap_dim()
        else:
            use_heatmap = False

        # local search
        solved_tours = list()
        for idx in range(tours.shape[0]):
            solved_tour = self._local_search(
                self.tours[idx], 
                self.points[idx // per_tours_num // per_heatmap_num],
                self.heatmap[idx // per_tours_num] if use_heatmap else None
            )
            solved_tours.append(solved_tour)                
        solved_tours = np.array(solved_tours)
        
        # check the tours
        for tour in solved_tours:
            if not self.is_valid_tour(tour):
                raise ValueError(f"The tour {tour} is not valid!")
            if tour[-1] != 0:
                raise ValueError(f"The tour {tour} is not valid!")
        
        return solved_tours
    
    def _local_search(
        self, tour: np.ndarray, heatmap: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """
        tour (np.ndarray): (nodes_num + 1,)
        heatmap (np.ndarray): (nodes_num, nodes_num,)
        points (np.ndarray): (nodes_num, 2)
        """
        raise NotImplementedError("_local_search is required to implemented in subclasses.")