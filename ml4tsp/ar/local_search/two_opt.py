import time
import torch
import numpy as np
from typing import Union
from ml4co_kit import to_tensor, to_numpy
from ml4tsp.ar.local_search.base import ML4TSPARLocalSearch


class ML4TSPARTwoOpt(ML4TSPARLocalSearch):
    def __init__(
        self,
        time_limit: Union[float, str] = "auto",
        max_iterations_2opt: int = 5000,
    ):
        super().__init__(time_limit=time_limit)
        self.max_iterations_2opt = max_iterations_2opt
    
    def _local_search(
        self, tour: np.ndarray, points: np.ndarray, heatmap: np.ndarray = None
    ) -> np.ndarray:
        # time_limit
        self.nodes_num = points.shape[-2]
        time_limit_flag = False if self.time_limit == "auto" else True
        
        # 2opt
        iterator = 0 
        begin_time = time.time()
        with torch.inference_mode():
            tour = to_tensor(tour).unsqueeze(dim=0).to(self.device)
            points = to_tensor(points).to(self.device)
            
            min_change = -1.0
            batch_size = 1
            while min_change < 0.0:
                points_i = points[tour[:, :-1].reshape(-1)].reshape(batch_size, -1, 1, 2)
                points_j = points[tour[:, :-1].reshape(-1)].reshape(batch_size, 1, -1, 2)
                points_i_plus_1 = points[tour[:, 1:].reshape(-1)].reshape(batch_size, -1, 1, 2)
                points_j_plus_1 = points[tour[:, 1:].reshape(-1)].reshape(batch_size, 1, -1, 2)
                
                A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
                A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
                A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
                A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))
                
                change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
                valid_change = torch.triu(change, diagonal=2)

                min_change = torch.min(valid_change)
                flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
                min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
                min_j = torch.remainder(flatten_argmin_index, len(points))

                if min_change < -1e-6:
                    for i in range(batch_size):
                        tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                    iterator += 1
                else:
                    break
                
                if iterator >= self.max_iterations_2opt:
                    break
            
                if time_limit_flag:
                    if time.time() - begin_time > self.time_limit:
                        break
                
        return to_numpy(tour[0])