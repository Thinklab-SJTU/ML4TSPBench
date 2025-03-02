import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from ml4co_kit import (
    TSPSolver, SOLVER_TYPE, Timer, sparse_points, TSPEvaluator,
    points_to_distmat, to_tensor, iterative_execution
)
from ..model.diffusion import ML4TSPDiffusion, InferenceSchedule


class ML4TSPT2TSolver(TSPSolver):
    def __init__(
        self, 
        model: ML4TSPDiffusion, 
        scale: int = 1, 
        seed: int = 1234,
        use_softdict: bool = False, # Rethink MCTS for TSP 
        softdict_tau: float = 0.0066,
        rewrite_steps: int = 3,
        rewrite_ratio: float = 0.3,
        steps_inf: int = 10
    ):
        super(ML4TSPT2TSolver, self).__init__(
            solver_type=SOLVER_TYPE.ML4TSP, scale=scale
        )
        self.model = model
        self.model.set_mode(mode="solve")
        torch.manual_seed(seed=seed)
        self.use_softdict = use_softdict
        self.softdict_tau = softdict_tau
        self.rewrite_steps = rewrite_steps
        self.rewrite_ratio = rewrite_ratio
        self.steps_inf = steps_inf
        
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
        evaluators = [TSPEvaluator(points[idx], self.norm) for idx in range(points.shape[0])]
        np_points = points.copy()
        
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
        
        # gradient search
        g_best_tours = solved_tours
        g_best_solved_costs = [evaluators[idx].evaluate(solved_tours[idx]) 
                               for idx in range(solved_tours.shape[0])]
        
        for _ in range(self.rewrite_steps):
            g_stacked_tours = []
            g_x0 = self.tour2adj(g_best_tours, np_points, edge_index.cpu()).to(device) 
            if sparse:
                g_x0 = g_x0.reshape(-1)
            g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()
            
            steps_T = int(self.model.diffusion_steps * self.rewrite_ratio)
            time_schedule = InferenceSchedule(inference_schedule=self.model.inference_schedule,
                                                T=steps_T, inference_T=self.steps_inf)
            
            Q_bar = torch.from_numpy(self.model.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)
            g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2]
            
            g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))  # [B, N, N]
            g_xt = g_xt * 2 - 1  # project to [-1, 1]
            g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise
            g_xt = (g_xt > 0).long()
            
            for i in range(self.steps_inf):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                if not sparse:
                    g_xt = self.model.guided_categorical_denoise_step(
                        points, g_xt, t1, device, edge_index, target_t=t2)
                else:
                    g_xt = self.model.guided_categorical_denoise_step(
                        points.reshape(-1, 2), g_xt, t1, device, 
                        edge_index.transpose(1, 0).reshape(2, -1), target_t=t2
                    )

            g_heatmap = g_xt.float().cpu().detach().numpy() + 1e-6

            # decode
            g_solved_tours, g_heatmap, per_tours_num = self.model.decoder.decode(
                heatmap=g_heatmap, points=points, edge_index=edge_index, 
                per_heatmap_num=per_heatmap_num
            ) # (B, P, N+1) & (B, P, N, N)
            g_solved_tours = g_solved_tours.reshape(-1, self.nodes_num + 1)
        
            # local search
            if self.model.local_search is not None:
                g_solved_tours = self.model.local_search.local_search(
                    tours=g_solved_tours, points=points, heatmap=g_heatmap, 
                    per_tours_num=per_tours_num, per_heatmap_num=per_heatmap_num
                )
                
            g_stacked_tours.append(g_solved_tours)
            g_solved_tours = np.concatenate(g_stacked_tours, axis=0)
            g_solved_costs = [evaluators[idx].evaluate(g_solved_tours[idx]) 
                                   for idx in range(g_solved_tours.shape[0])]

            for i in range(g_best_tours.shape[0]):
                if g_solved_costs[i] < g_best_solved_costs[i]:
                    g_best_tours[i, :] = g_solved_tours[i]
                    g_best_solved_costs[i] = g_solved_costs[i]
            
        return g_best_tours
    
    def tour2adj(self, tours, points, edge_index):
        sparse = self.model.env.sparse
        sparse_factor = self.model.env.sparse_factor
        batch_size = points.shape[0]
    
        if not sparse:
            adj_matrices = torch.zeros((batch_size, points.shape[1], points.shape[1]))
            for idx in range(batch_size):
                for j in range(tours[idx].shape[0] - 1):
                    adj_matrices[idx, tours[idx][j], tours[idx][j + 1]] = 1
                    # adj_matrices[idx, tours[idx][j + 1], tours[idx][j]] = 1
        else:
            adj_matrices = torch.zeros((batch_size, edge_index.shape[2]))
            for idx in range(batch_size):
                target1 = np.zeros(points.shape[1], dtype=np.int64)
                target2 = np.zeros(points.shape[1], dtype=np.int64)
                target1[tours[idx][:-1]] = tours[idx][1:]
                target2[tours[idx][1:]] = tours[idx][:-1]
                target1 = torch.from_numpy(target1).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
                target2 = torch.from_numpy(target2).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
                
                adj_matrix = torch.eq(edge_index[idx][1], target1) #+ torch.eq(edge_index[idx][1], target2)
                adj_matrices[idx, :] = adj_matrix.to(torch.int)
        return adj_matrices