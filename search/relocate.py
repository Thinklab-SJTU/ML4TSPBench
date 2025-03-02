import numpy as np
import itertools
import scipy.spatial as ssp

def relocate_cost(tour, dist_mat, i, j):
    if i == j:
        return 0
    a, b, c = tour[i - 1], tour[i], tour[i + 1]
    if i < j:
        d, e = tour[j], tour[j + 1]
    else:
        d, e = tour[j - 1], tour[j]
    delta = - dist_mat[a, b] - dist_mat[b, c] + dist_mat[a, c] \
            - dist_mat[d, e] + dist_mat[d, b] + dist_mat[b, e]
    return delta

def relocate(tour, i, j):
    new_tour = tour.copy().tolist()
    n = new_tour.pop(i)
    new_tour.insert(j, n)
    return new_tour

def get_relocate_tour(tour, dist_mat):
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)

    for i, j in itertools.permutations(idxs, 2):
        if i - j == 1:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
            continue
        delta = relocate_cost(tour, dist_mat, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour

def tsp_relocate(np_points: np.ndarray, tours, adj_mat=None, device="cpu", **kwargs):
    '''
    Output: tours, shape (parallel_sampling, N + 1)
    '''
    # dimension modification
    nodes_num = adj_mat.shape[-1]
    initial_dim = adj_mat.ndim
    if adj_mat.ndim == 2:
        adj_mat = np.expand_dims(adj_mat, axis=0)
    if np_points.ndim == 2:
        np_points = np.expand_dims(np_points, axis=0)
    tours = np.array(tours)
    if tours.ndim == 1:
        tours = np.expand_dims(tours, axis=0)
    solved_tours = []
    
    for i in range(adj_mat.shape[0]):
        dist_mat = ssp.distance_matrix(np_points[i], np_points[i])
        _, tour = get_relocate_tour(tours[i], dist_mat)
        solved_tours.append(tour)
    
    return solved_tours