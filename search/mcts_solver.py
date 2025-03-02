from multiprocessing import Pool
import numpy as np
import scipy.sparse
import scipy.spatial
import ctypes
import numpy as np
from .c_mcts_solver import c_mcts_solver
from typing import Optional
from .mcts_smooth import smooth_heatmap, smooth_heatmap_v2


def _tsp_mcts_solver(
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    kwargs: dict
):
    # gain the params from the kwargs dict
    mcts_max_depth = kwargs.get('mcts_max_depth', 10)
    mcts_param_t = kwargs.get('mcts_param_t', 0.1)
    mcts_smooth = kwargs.get('mcts_smooth', False)
    mcts_smooth_v2 = kwargs.get('mcts_smooth_v2', False)
    max_iterations_2opt = kwargs.get('max_iterations_2opt', 5000)
    
    # dimension modification
    nodes_num = adj_mat.shape[-1]
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
    if np_points.ndim == 2:
        np_points = np.expand_dims(np_points, axis=0)
        
    # mcts solver
    shortest_tours = list()
    for i in range(adj_mat.shape[0]):
        heatmap = adj_mat[i]
        if mcts_smooth:
            heatmap = smooth_heatmap(heatmap)
        elif mcts_smooth_v2:
            heatmap = smooth_heatmap_v2(heatmap, np_points[i])
        heatmap = heatmap.reshape(-1)
        nodes_coords = np_points[i].reshape(-1)
        tour = c_mcts_solver(
            heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
            nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
            nodes_num,
            mcts_max_depth,
            ctypes.c_float(mcts_param_t),
            max_iterations_2opt
        )
        tour = np.ctypeslib.as_array(tour, shape=(nodes_num,))
        tour = tour.tolist()
        tour.append(tour[0])
        shortest_tours.append(tour)
    
    # dimension modification
    if adj_mat.shape[0] == 1:
        shortest_tours = shortest_tours[0]
    return shortest_tours


def tsp_mcts_solver(
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    edge_index_np: Optional[np.ndarray]=None, 
    sparse_graph=False, 
    parallel_sampling=1,
    **kwargs
):
    '''
    Output: tours, shape (parallel_sampling, N + 1)
    Reference: https://github.com/Spider-scnu/TSP
    '''
    # ensure the dtype of the np.ndarray
    np_points = np_points.astype(np.float32)
    adj_mat = adj_mat.astype(np.float32)

    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    if not sparse_graph:
        splitted_adj_mat = [
            (adj_mat[0] + adj_mat[0].T)/2 for adj_mat in splitted_adj_mat
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
    spliited_kwargs = [kwargs for _ in range(parallel_sampling)]
    
    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                _tsp_mcts_solver,
                zip(splitted_adj_mat, splitted_points, spliited_kwargs),
            )
    else:
        results = [
            _tsp_mcts_solver(_adj_mat, _np_points, _args) for _adj_mat, _np_points, _args \
                in zip(splitted_adj_mat, splitted_points, spliited_kwargs)
        ]
        
    return results
