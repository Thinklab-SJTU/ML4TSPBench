import torch
import numpy as np
from tqdm import tqdm
import lkh
import time
import warnings
import tsplib95
from solvers.pyconcorde import TSPSolver as _TSPConcordeSolver
from typing import Union, Any
from pytorch_lightning.utilities import rank_zero_info
from models.modelzoo import TSPNAREncoder, TSPAREncoder, get_nar_model, get_ar_model
from search import get_decoding_func, get_local_search_func
from utils.utils import generate_tsp_file, generate_opt_tour_file, get_data_from_tsp_file, \
    sparse_points, get_tour_from_tour_file
from utils.evaluator import TSPEvaluator
from torch_sparse import SparseTensor

warnings.filterwarnings("ignore")


class TSPSolver:
    """
    """
    def __init__(self):
        self.points = None
        self.ori_points = None
        self.edge_index = None
        self.sparse = False
        self.tours = None
        self.gt_tours = None
        self.num_nodes = None

    def check_points_dim(self):
        if self.points is None:
            return
        elif self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        self.num_nodes = self.points.shape[1]

    def from_tspfile(self, filename: str):
        assert filename.endswith(".tsp"), f"file name error"
        data = get_data_from_tsp_file(filename)
        points = data.node_coords
        if points is None:
            raise RuntimeError("Error in loading {}".format(filename))
        points = np.array(points)
        self.ori_points = points
        self.points = points.astype(np.float32)
        self.check_points_dim()

    def from_txt(self, filename: str, load_gt_tours: str=True):
        assert filename.endswith(".txt"), f"file name error"
        with open(filename, 'r') as file:
            nodes_coords = list()
            gt_tours = list()
            for line in file:
                line = line.strip()
                if 'output' in line:
                    split_line = line.split(' output ')
                    points = split_line[0]
                    tour = split_line[1]
                    tour = tour.split(' ')
                    tour = np.array([int(t) for t in tour])
                    tour -= 1
                    gt_tours.append(tour)
                else:
                    points = line
                    load_gt_tours = False
                points = points.split(' ')
                points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
                nodes_coords.append(points)
        
        if load_gt_tours:
            self.gt_tours = np.array(gt_tours)
        nodes_coords = np.array(nodes_coords)
        self.ori_points = nodes_coords
        self.points = nodes_coords.astype(np.float32)
        self.check_points_dim()

    def from_data(self, points: np.ndarray):
        self.points = points.astype(np.float32)
        self.check_points_dim()
    
    def read_gt_tours_from_tspfile(self, filename: str):
        assert filename.endswith(".opt.tour"), f"file name error"
        self.gt_tours = get_tour_from_tour_file(filename)
    
    def to_tsp_file(
        self,
        tour_filename: str=None,
        filename: str=None,
        points: Union[np.ndarray, list]=None,
        tours: Union[np.ndarray, list]=None,

    ):
        # points
        if points is None:
            points = self.points
        if type(points) == list:
            points = np.array(points)
        # tours
        if tours is None:
            tours = self.tours
        if type(tours) == list:
            tours = np.array(tours)
        # generate .tsp file if filename is not none
        if filename is not None:
            assert filename.endswith('.tsp')
            generate_tsp_file(points, filename)
        # generate .tsp.tour file if tour_filename is not none
        if tour_filename is not None:
            assert filename.endswith('.tsp.tour')
            generate_opt_tour_file(tours, filename)
            
    def to_txt(
        self, 
        filename: str="example.txt",
        points: Union[np.ndarray, list]=None,
        tours: Union[np.ndarray, list]=None,
    ):
        # points
        if points is None:
            points = self.ori_points
        if type(points) == list:
            points = np.array(points)
        # tours
        if tours is None:
            tours = self.tours
        if type(tours) == list:
            tours = np.array(tours)
        # write
        with open(filename, "w") as f:
            for idx, tour in enumerate(tours):
                f.write(" ".join(str(x) + str(" ") + str(y) for x, y in points[idx]))
                f.write(str(" ") + str('output') + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                f.write("\n")
            f.close()

    def evaluate(self, caculate_gap: bool=False):
        """
        """
        if caculate_gap and self.gt_tours is None:
            raise ValueError("gt_tours cannot be None, please use TSPLKHSolver to get the gt_tours.")
        if self.tours is None:
            raise ValueError("tours cannot be None, please use method 'solve' to solve solution.")
        
        if self.points.ndim == 2:
            evaluator = TSPEvaluator(self.points)
            solved_cost = evaluator.evaluate(self.tours)
            return solved_cost
        else:
            cost_total = 0
            batch = self.points.shape[0]
            if caculate_gap:
                gap_list = list()
            if self.tours.ndim != batch:
                tours = self.tours.reshape(batch, -1, self.tours.shape[-1])
                for idx in range(batch):
                    evaluator = TSPEvaluator(self.points[idx])
                    solved_tours = tours[idx]
                    solved_costs = list()
                    for tour in solved_tours:
                        solved_costs.append(evaluator.evaluate(tour))
                    solved_cost = np.min(solved_costs)
                    cost_total += solved_cost
                    if caculate_gap:
                        gt_cost = evaluator.evaluate(self.gt_tours[idx])
                        gap = (solved_cost - gt_cost) / gt_cost * 100
                        gap_list.append(gap)
            else:
                for idx in range(batch):
                    evaluator = TSPEvaluator(self.points[idx])
                    solved_tour = self.tours[idx]
                    solved_cost = evaluator.evaluate(solved_tour)
                    cost_total += solved_cost
                    if caculate_gap:
                        gt_cost = evaluator.evaluate(self.gt_tours[idx])
                        gap = (solved_cost - gt_cost) / gt_cost * 100
                        gap_list.append(gap)
            cost_avg = cost_total / batch
            if caculate_gap:
                gap_avg = np.sum(gap_list) / batch
                gap_std = np.std(gap_list)
                return cost_avg, gap_avg, gap_std
            else:
                return cost_avg


class TSPConcordeSolver(TSPSolver):
    """
    """
    def __init__(self, concorde_scale: int=1e6):
        super(TSPConcordeSolver, self).__init__()
        self.concorde_scale = concorde_scale

    def _solve(self, nodes_coord: np.ndarray):
        solver = _TSPConcordeSolver.from_data(nodes_coord[:, 0] * self.concorde_scale, 
                nodes_coord[:, 1] * self.concorde_scale, norm="GEO")
        solution = solver.solve(verbose=False)
        tour = solution.tour
        return tour
        
    def solve(self):
        start_time = time.time()
        if self.points is None:
            raise ValueError("points is None!")
        tours = list()
        num_points = self.points.shape[0]
        for idx in tqdm(range(num_points), desc="Solving TSP Using Concorde"):
            tours.append(self._solve(self.points[idx]))
        self.tours = np.array(tours)
        zeros = np.zeros((self.tours.shape[0], 1))
        self.tours = np.append(self.tours, zeros, axis=1).astype(np.int32)
        end_time = time.time()
        print(f"Use Time: {end_time - start_time}")
        return self.tours


class TSPLKHSolver(TSPSolver):
    """
    """
    def __init__(self, lkh_path: str="LKH", lkh_scale: int=1e6):
        super(TSPLKHSolver, self).__init__()
        self.lkh_path = lkh_path
        self.lkh_scale = lkh_scale

    def _solve(self, nodes_coord: np.ndarray, max_trials: int):
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = self.num_nodes
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {n + 1: nodes_coord[n] * \
            self.lkh_scale for n in range(self.num_nodes)}
        solution = lkh.solve(self.lkh_path, problem=problem, max_trials=max_trials, runs=10)
        tour = [n - 1 for n in solution[0]]
        tour.append(0)
        return tour
        
    def solve(self, max_trials: int=1000):
        start_time = time.time()
        if self.points is None:
            raise ValueError("points is None!")
        tours = []
        num_points = self.points.shape[0]
        for idx in tqdm(range(num_points), desc="Solving TSP Using LKH"):
            tours.append(self._solve(self.points[idx], max_trials))
        self.tours = np.array(tours)
        end_time = time.time()
        print(f"Use Time: {end_time - start_time}")
        return self.tours


class TSPNARSolver(TSPSolver):
    """
    """
    def __init__(self):
        super(TSPNARSolver, self).__init__()

    def solve(
        self,
        batch_size: int=16,
        sparse_factor: int=-1,
        encoder: Union[TSPNAREncoder, str] = "gnn",
        encoder_kwargs: dict = {},
        decoding_type: Union[Any, str] = "greedy",
        decoding_kwargs: dict = {},
        local_search_type: str = None,
        ls_kwargs: dict = {},
        active_search: bool = False,
        pretrained: bool=True,
        device='cpu',
    ):
        self.sparse = sparse_factor > 0
        self.active_search = active_search
        self.decoding_type = decoding_type
        self.ls_type = local_search_type

        # encoder & gain heatmap
        if type(encoder) == str:
            encoder_kwargs.update({
                "mode": "solve", 
                "sparse_factor": sparse_factor,
                "num_nodes": self.num_nodes
                }
            )
            self.encoder = get_nar_model(task="tsp", name=encoder)(**encoder_kwargs)
        else:
            self.encoder = encoder
        rank_zero_info(f"Begin encoding, Using {self.encoder}")
        if pretrained:
            rank_zero_info(f"Loading Weights from Pretrained CheckPoint")
            ckpt_path = encoder_kwargs["ckpt_path"] if "ckpt_path" in encoder_kwargs.keys() else None
            self.encoder.load_ckpt(ckpt_path)
        self.encoder.to(device)

        solve_begin_time = time.time()
        edge_index = None
        if self.sparse:
            points, edge_index = sparse_points(self.points, sparse_factor, device)
            np_edge_index = edge_index.detach().cpu().numpy()
        else:
            points = self.points
        heatmap = self.encoder.solve(points, edge_index, batch_size, device)
        solve_end_time = time.time()
        solve_time = solve_end_time - solve_begin_time
        rank_zero_info(f"Model Solve, Using {solve_time}")

        # get  sparse to dense time
        # s2d_begin_time = time.time()
        # new_heatmaps = list()
        # for idx in range(heatmap.shape[0]):
        #     sparse_adj_mat = SparseTensor(
        #         row=edge_index[idx][0],
        #         col=edge_index[idx][1],
        #         value=torch.tensor(heatmap[idx]).to(device=edge_index.device)
        #     )
        #     adj_mat = sparse_adj_mat.to_dense().cpu().numpy()
        #     new_heatmaps.append(adj_mat)
        # new_heatmaps = np.array(new_heatmaps)
        # np.squeeze(new_heatmaps,axis=1)
        # s2d_end_time = time.time()
        # s2d_time = s2d_end_time - s2d_begin_time
        # rank_zero_info(f"Sparse to Dense, Using {s2d_time}")
   
        # decoding
        decoding_kwargs.update({"sparse_factor": sparse_factor})
        if type(decoding_type) == str:
            self.decoding_func = get_decoding_func(task="tsp", name=decoding_type)
        else:
            self.decoding_func = decoding_type
        rank_zero_info(f"Begin Decoding, Using {self.decoding_func.__name__}")
        decoded_tours = list()
        for idx in tqdm(range(self.points.shape[0]), desc='Decoding'):
            if not self.sparse:
                adj_mat = np.expand_dims(heatmap[idx], axis=0)
            else:
                adj_mat = heatmap[idx]
            tour = self.decoding_func(
                adj_mat=adj_mat, 
                np_points=self.points[idx], 
                edge_index_np=np_edge_index[idx] if self.sparse else None,
                sparse_graph=self.sparse,
                **decoding_kwargs
            )
            decoded_tours.append(tour[0])
        decoded_tours = np.array(decoded_tours)
        
        # local_search
        ls_tours = None
        self.local_search_func = get_local_search_func(task="tsp", name=local_search_type)        
        if self.local_search_func is not None:
            rank_zero_info(f"Begin Local Search, Using {self.local_search_func.__name__}")
            ls_tours = list()
            for idx in tqdm(range(self.points.shape[0]), desc='Local Search'):
                adj_mat = heatmap[idx]
                if self.sparse:
                    sparse_adj_mat = SparseTensor(
                        row=edge_index[idx][0].long(),
                        col=edge_index[idx][1].long(),
                        value=torch.tensor(adj_mat).to(device=edge_index.device)
                    )
                    adj_mat = sparse_adj_mat.to_dense().unsqueeze(dim=0).cpu().numpy()
                tour = self.local_search_func(
                    np_points=self.points[idx],
                    tours=decoded_tours[idx],
                    adj_mat=adj_mat,
                    device=device,
                    **ls_kwargs
                )
                ls_tours.append(tour)
            ls_tours = np.array(ls_tours)

        tours = decoded_tours if ls_tours is None else ls_tours
        self.tours = tours
        return tours

    def __repr__(self):
        message = f"encoder={self.encoder}, decoding_type={self.decoding_type}, ls_type={self.ls_type}"
        return f"{self.__class__.__name__}({message})"
    

class TSPARSolver(TSPSolver):
    """
    """
    def __init__(self):
        super(TSPARSolver, self).__init__()

    def solve(
        self,
        batch_size: int=16,
        encoder: Union[TSPAREncoder, str] = "am",
        encoder_kwargs: dict = {},
        decoding_type: Union[Any, str] = "greedy",
        decoding_kwargs: dict = {},
        ls_type: str = None,
        ls_kwargs: dict = {},
        active_search: bool = False,
        pretrained: bool=True,
        device='cpu',
    ):
        self.active_search = active_search
        self.decoding_type = decoding_type
        self.ls_type = ls_type

        # encoder & gain heatmap
        if type(encoder) == str:
            encoder_kwargs.update({"mode": "solve"})
            self.encoder = get_ar_model(task="tsp", name=encoder)(
                decoding_type=decoding_type,
                **encoder_kwargs,
                **decoding_kwargs,
            )
        else:
            self.encoder = encoder
        rank_zero_info(f"Begin solving, Using {self.encoder}")
        if pretrained:
            rank_zero_info(f"Loading Weights from Pretrained CheckPoint")
            self.encoder.load_ckpt()
        self.encoder.to(device)
        np_tours = self.encoder.solve(self.points, batch_size=batch_size, device=device)

        # local_search
        ls_tours = None
        self.local_search_func = get_local_search_func(task="tsp", name=ls_type)        
        if self.local_search_func is not None:
            rank_zero_info(f"Begin Local Search, Using {self.local_search_func.__name__}")
            ls_tours = list()
            for idx in tqdm(range(self.points.shape[0]), desc='Local Search'):
                tour = self.local_search_func(
                    np_points=self.points[idx],
                    tours=np_tours[idx],
                    adj_mat=None,
                    device=device,
                    **ls_kwargs
                )
                ls_tours.append(tour)
            ls_tours = np.array(ls_tours)

        tours = np_tours if ls_tours is None else ls_tours
        self.tours = tours
        return tours
    
    def __repr__(self):
        message = f"encoder={self.encoder}, decoding_type={self.decoding_type}, ls_type={self.ls_type}"
        return f"{self.__class__.__name__}({message})"
