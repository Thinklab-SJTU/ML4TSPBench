from multiprocessing import Pool
import numpy as np
import scipy.sparse
import scipy.spatial
import ctypes
from .c_mcts import c_mcts
from .mcts_solver import smooth_heatmap
from models.utils.evaluator import TSPEvaluator
from typing import Optional
from .mcts_smooth import smooth_heatmap, smooth_heatmap_v2


def generate_random_array(nodes_num: int):
    arr = np.arange(1, nodes_num)
    np.random.shuffle(arr)
    result = np.insert(arr, [0, len(arr)], [0, 0])
    return result


def _tsp_rmcts(adj_mat: np.ndarray, np_points: np.ndarray, kwargs:dict):
    """
    """
    # gain the params from the kwargs dict
    samples = kwargs.get("random_samples", 10)
    mcts_max_depth = kwargs.get('mcts_max_depth', 10)
    max_iterations = kwargs.get('max_iterations', 5000)
    mcts_param_t = kwargs.get('mcts_param_t', 0.1)
    mcts_smooth_v2 = kwargs.get('mcts_smooth_v2', False)
    mcts_smooth = kwargs.get('mcts_smooth', False)

    # dimension modification
    nodes_num = adj_mat.shape[-1]
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
        np_points = np.expand_dims(np_points, axis=0)
    
    shortest_tours = list()

    for i in range(adj_mat.shape[0]):
        heatmap = adj_mat[i]
        if mcts_smooth:
            heatmap = smooth_heatmap(heatmap)
        elif mcts_smooth_v2:
            heatmap = smooth_heatmap_v2(heatmap, np_points[i])
        nodes_coords = np_points[i].reshape(-1)
        samples_tours = list()
        for _ in range(samples):
            tour = generate_random_array(nodes_num)
            np_tour = np.array(tour, dtype=np.int16)
            mcts_tour = c_mcts(
                np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
                heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                nodes_num,
                mcts_max_depth,
                ctypes.c_float(mcts_param_t),
                max_iterations,
            )
            mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
            mcts_tour = mcts_tour.tolist()
            mcts_tour.append(mcts_tour[0])
            samples_tours.append(mcts_tour)
             
        tsp_solver = TSPEvaluator(np_points[0])
        best_tour = None
        best_tour_cost = 1e10

        for tour in samples_tours:
            tour_cost = tsp_solver.evaluate(tour)
            if tour_cost < best_tour_cost:
                best_tour_cost = tour_cost
                best_tour = tour
        shortest_tours.append(best_tour)
 
    if adj_mat.shape[0] == 1:
        shortest_tours = shortest_tours[0]

    return shortest_tours


def tsp_rmcts(
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    edge_index_np: Optional[np.ndarray]=None, 
    sparse_graph: bool=False, 
    parallel_sampling: int=1,
    **kwargs
):
    '''
    '''
    # ensure the dtype of the np.ndarray
    np_points = np_points.astype(np.float32)
    adj_mat = adj_mat.astype(np.float32)
    
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    if not sparse_graph:
        splitted_adj_mat = [
            (adj_mat[0] + adj_mat[0].T) / 2 for adj_mat in splitted_adj_mat
        ]
    else:
        if edge_index_np is None:
            raise ValueError("edge_index_np should be given if sparse_graph is True")
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (adj_mat/2, (edge_index_np[0], edge_index_np[1])),
            ).toarray() + scipy.sparse.coo_matrix(
                (adj_mat/2, (edge_index_np[1], edge_index_np[0])),
            ).toarray() for adj_mat in splitted_adj_mat
        ]
        
    splitted_points = [np_points for _ in range(parallel_sampling)]
    spliited_args = [kwargs for _ in range(parallel_sampling)]
    
    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                _tsp_rmcts,
                zip(splitted_adj_mat, splitted_points, spliited_args),
            )
    else:
        results = [
            _tsp_rmcts(_adj_mat, _np_points, _args) for _adj_mat, _np_points, _args \
                in zip(splitted_adj_mat, splitted_points, spliited_args)
        ]
        
    return results
