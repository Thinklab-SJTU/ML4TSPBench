from multiprocessing import Pool
import numpy as np
import torch
from scipy.spatial.distance import cdist
import scipy.sparse
import scipy.spatial
from .beam_search import Beamsearch, tour_nodes_to_tour_len, is_valid_tour
import ctypes
from .c_mcts import c_mcts
from .mcts_smooth import smooth_heatmap, smooth_heatmap_v2


def _tsp_beam_mcts(
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    kwargs: dict
):
    '''
    tsp beam mcts solver
    '''
    # gain the params from the kwargs dict
    beam_size = kwargs.get('beam_size', 50)
    probs_type = kwargs.get('beam_probs_type', 'raw')
    beam_random_smart = kwargs.get('beam_random_smart', False)
    mcts_param_t = kwargs.get('mcts_param_t', 0.1)
    mcts_smooth = kwargs.get('mcts_smooth', False)
    mcts_smooth_v2 = kwargs.get('mcts_smooth_v2', False)
    max_iterations_2opt = kwargs.get('max_iterations_2opt', 5000)
    mcts_max_depth = kwargs.get('mcts_max_depth', 10)
    
    # dimension modification
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
        batch_size, num_nodes, _ = adj_mat.shape
        x_edges_values = np.expand_dims(cdist(np_points, np_points), axis=0)
    else:
        batch_size, num_nodes, _ = adj_mat.shape
        x_edges_values = np.array([cdist(coords, coords) for coords in np_points])
    if np_points.ndim == 2:
        np_points = np.expand_dims(np_points, axis=0)
    
    # beam search
    nodes_num = adj_mat.shape[-1]
    beamsearch = Beamsearch(
        beam_size=beam_size, 
        batch_size=batch_size, 
        num_nodes=num_nodes,
        probs_type=probs_type, 
        random_start=beam_random_smart
    )
    adj_mat = torch.tensor(adj_mat)
    if probs_type == 'logits':
        beam_mat = torch.log(adj_mat) - 1e-10
    else:
        beam_mat = adj_mat
    trans_probs = beam_mat.gather(1, beamsearch.get_current_state())
    for _ in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = beam_mat.gather(1, beamsearch.get_current_state())
    ends = torch.zeros(batch_size, 1).long()
    shortest_tours = beamsearch.get_hypothesis(ends)
    shortest_lens = torch.zeros(batch_size, 1)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx])

    # mcts
    for pos in range(0, beam_size):
        ends = pos * torch.ones(batch_size, 1).long() # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for batch in range(batch_size):
            heatmap = adj_mat[batch].cpu().detach().numpy()
            if mcts_smooth:
                heatmap = smooth_heatmap(heatmap)
            elif mcts_smooth_v2:
                heatmap = smooth_heatmap_v2(heatmap, np_points[batch])
            heatmap = heatmap.reshape(-1)
            tour = hyp_tours[batch]
            tour = tour.tolist()
            tour.append(0)            
            np_tour = np.array(tour, dtype=np.int16)
            nodes_coords = np_points[batch].reshape(-1)
            mcts_tour = c_mcts(
                np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
                heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                nodes_num,
                mcts_max_depth,
                ctypes.c_float(mcts_param_t),
                max_iterations_2opt
            )
            mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
            mcts_tour = mcts_tour.tolist()
            hyp_len = tour_nodes_to_tour_len(mcts_tour, x_edges_values[batch])
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[batch] and is_valid_tour(mcts_tour, num_nodes):
                shortest_tours[batch] = torch.tensor(mcts_tour)
                shortest_lens[batch] = hyp_len

    # format modification
    shortest_tours = torch.cat([shortest_tours, 
            shortest_tours[:, 0].reshape(batch_size, 1)], dim=1)
    if batch_size == 1:
        shortest_tours = shortest_tours.squeeze(dim=0)
    shortest_tours = shortest_tours.tolist()

    return shortest_tours


def tsp_beam_mcts(adj_mat: np.ndarray, np_points: np.ndarray, edge_index_np: np.ndarray=None, 
                  
             sparse_graph=False, parallel_sampling=1, device="cpu", **kwargs):
    """_summary_

    Args:
        adj_mat (np.ndarray): the predict heatmap (B, N, N)
        np_points (np.ndarray): the coords of nodes (B, N, 2)
        edge_index_np (np.ndarray, optional): the edge_index for sparse heatmap. Defaults to None.
        sparse_graph (bool, optional): whether the graph is sparse. Defaults to False.
        parallel_sampling (int, optional): _description_. Defaults to 1.
        device (str, optional): _description_. Defaults to "cpu".
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # ensure the dtype of the np.ndarray
    np_points = np_points.astype(np.float32)
    adj_mat = adj_mat.astype(np.float32)

    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    if not sparse_graph:
        splitted_adj_mat = [
            (adj_mat[0] + adj_mat[0].T)/2 for adj_mat in splitted_adj_mat
        ]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (adj_mat/2, (edge_index_np[0], edge_index_np[1])),
            ).toarray() + scipy.sparse.coo_matrix(
                (adj_mat/2, (edge_index_np[1], edge_index_np[0])),
            ).toarray() + 1e-14 for adj_mat in splitted_adj_mat
        ]
    
    splitted_points = [np_points for _ in range(parallel_sampling)]
    spliited_kwargs = [kwargs for _ in range(parallel_sampling)]
    
    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                _tsp_beam_mcts,
                zip(splitted_adj_mat, splitted_points, spliited_kwargs),
            )
    else:
        results = [
            _tsp_beam_mcts(_adj_mat, _np_points, _args) for _adj_mat, _np_points, _args \
                in zip(splitted_adj_mat, splitted_points, spliited_kwargs)
        ]

    return results
