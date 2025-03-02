import numpy as np
from ml4co_kit import tsp_mcts_local_search
from .mcts_smooth import smooth_heatmap, smooth_heatmap_v2


MCTS_SETTINGS = {
      50: (0.005, 2, False, 5000),
     100: (0.020, 2, False, 50),
     200: (0.100, 1, True, 0),
     500: (1.000, 1, True, 0),
    1000: (5.000, 1, True, 0)
}


def get_nodes_num_for_mcts_time_limit(x: int):
    if x <= 75:
        return 50
    if x <= 150:
        return 100
    if x <= 300:
        return 200
    if x <= 600:
        return 500
    else:
        return 1000
    

def tsp_mcts(
    np_points:np.ndarray, 
    tours: np.ndarray, 
    adj_mat: np.ndarray, 
    **kwargs
):
    '''
    Output: tours, shape (parallel_sampling, N + 1)
    '''
    assert adj_mat is not None, "adj_mat(heatmap) must be given in mcts."
    # gain the params from the kwargs dict
    mcts_max_depth = kwargs.get('mcts_max_depth', 10)
    max_iterations_2opt = kwargs.get('mcts_max_iterations_2opt', None)
    time_limit = kwargs.get('mcts_time_limit', None)
    mcts_smooth = kwargs.get('mcts_smooth', False)
    mcts_smooth_v2 = kwargs.get('mcts_smooth_v2', False)
    
    # ensure the dtype of the np.ndarray
    np_points = np_points.astype(np.float32)
    adj_mat = adj_mat.astype(np.float32)
    
    # dimension modification
    nodes_num = adj_mat.shape[-1]
    initial_dim = adj_mat.ndim
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
    if np_points.ndim == 2:
        np_points = np.expand_dims(np_points, axis=0)
    if np_points.shape[0] != adj_mat.shape[0]:
        np_points = np_points.repeat(adj_mat.shape[0] / np_points.shape[0], axis=0)
    tours = np.array(tours)
    if tours.ndim == 1:
        tours = np.expand_dims(tours, axis=0)
    
    # settings
    x = get_nodes_num_for_mcts_time_limit(nodes_num)
    settings = MCTS_SETTINGS[x]
    _, type_2opt, continue_flag, _ = settings
    if time_limit is None:
        time_limit = settings[0]
    if max_iterations_2opt is None:
        max_iterations_2opt = settings[3]
        
    # mcts local search
    shortest_tours = list()    
    for i in range(adj_mat.shape[0]):
        heatmap = adj_mat[i]
        if mcts_smooth:
            heatmap = smooth_heatmap(heatmap)
        elif mcts_smooth_v2:
            heatmap = smooth_heatmap_v2(heatmap, np_points[i])
        nodes_coords = np_points[i]
        input_tour = np.array(tours[i], dtype=np.int16)
        tour = tsp_mcts_local_search(
            init_tours=input_tour, heatmap=heatmap, points=nodes_coords,
            time_limit=time_limit, max_depth=mcts_max_depth, continue_flag=continue_flag,
            type_2opt=type_2opt, max_iterations_2opt=max_iterations_2opt
        )
        shortest_tours.append(tour)
        
    # dimension modification
    if initial_dim == 2:
        shortest_tours = shortest_tours[0]
    return shortest_tours
