import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from ml4co_kit import (
    TSPSolver, SOLVER_TYPE, Timer, sparse_points, 
    points_to_distmat, to_tensor, iterative_execution
)
from ml4tsp.nar.model.base import ML4TSPNARBaseModel


class ML4TSPNARSolver(TSPSolver):
    def __init__(
        self, 
        model: ML4TSPNARBaseModel, 
        scale: int = 1, 
        seed: int = 1234,
        use_softdict: bool = False, # Rethink MCTS for TSP 
        softdict_tau: float = 0.0066
    ):
        super(ML4TSPNARSolver, self).__init__(
            solver_type=SOLVER_TYPE.ML4TSP, scale=scale
        )
        self.model = model
        self.model.set_mode(mode="solve")
        torch.manual_seed(seed=seed)
        self.use_softdict = use_softdict
        self.softdict_tau = softdict_tau
        
    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        batch_size: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        # preparation
        self.from_data(points=points, norm=norm, normalize=normalize)
        self.model.set_nodes_num(self.nodes_num)
        timer = Timer(apply=show_time)
        timer.start()
        
        # batch process
        samples_num = self.points.shape[0]
        iters = samples_num // batch_size
        batch_points = self.points.reshape(iters, batch_size, self.nodes_num, 2)
        
        # solve (core)
        tours_list = list()
        for idx in iterative_execution(range, iters, self.solve_msg, show_time):
            tours_list.append(self._solve(batch_points[idx]))
            
        # format
        tours = np.array(tours_list).reshape(-1, self.nodes_num + 1)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()  
                  
        # return
        return self.tours

    def _solve(self, points: np.ndarray) -> np.ndarray:
        # utils
        device = self.model.env.device
        sparse = self.model.env.sparse
        sparse_factor = self.model.env.sparse_factor
        
        # process data
        if sparse:
            points, edge_index = sparse_points(
                points=points, sparse_factor=sparse_factor, device=device
            )
            distmat = points_to_distmat(points, edge_index).to(device)    
        else:
            points, edge_index = to_tensor(points).to(device), None
            distmat = points_to_distmat(points, edge_index).to(device)
        
        # softdict
        if not self.use_softdict:
            heatmap = self.model.inference_process(
                points=points, edge_index=edge_index, distmat=distmat, ground_truth=None 
            )
        else:
            points = to_tensor(self.points)
            distance_matrix = points_to_distmat(points)
            eye = torch.eye(distance_matrix.size(1)).unsqueeze(0)
            distance_matrix = torch.where(
                eye == 1, torch.tensor(float('inf'), dtype=torch.float), distance_matrix
            )
            heatmap = F.softmax(- distance_matrix / self.softdict_tau, dim=2)
        per_heatmap_num = len(heatmap) // len(points)
            
        # decode
        solved_tours, heatmap, per_tours_num = self.model.decoder.decode(
            heatmap=heatmap, points=points, edge_index=edge_index, 
            per_heatmap_num=per_heatmap_num
        ) # (B, P, N+1) & (B, P, N, N)
        solved_tours = solved_tours.reshape(-1, self.nodes_num + 1)
    
        # local search
        if self.model.local_search is not None:
            solved_tours = self.model.local_search.local_search(
                tours=solved_tours, points=points, heatmap=heatmap, 
                per_tours_num=per_tours_num, per_heatmap_num=per_heatmap_num
            )

        return solved_tours