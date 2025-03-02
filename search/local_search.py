import numpy as np
import scipy
import itertools
import numpy as np
import time
from utils.evaluator import TSPEvaluator


def swap(tour, i, j):
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j - 1: i - 1: -1] + tour[j:]


def two_opt_cost(tour, dist_mat, i, j):
    if i == j:
        return 0
    elif j < i:
        i, j = j, i
    a, b, c, d = tour[i], tour[i - 1], tour[j], tour[j - 1]
    delta = dist_mat[a, c] + dist_mat[b, d] + dist_mat[a, b] - dist_mat[c, d]
    return delta


def get_two_opt_tour(tour, dist_mat, fixed_i=None):
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)

    if fixed_i is not None:
        i = fixed_i
        assert i > 0 and i < len(tour) - 1
        for j in idxs:
            if abs(i - j) < 2:
                continue
            delta = two_opt_cost(tour, dist_mat, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
    else:
        for i, j in itertools.combinations(idxs, 2):
            if abs(i - j) < 2:
                continue
            delta = two_opt_cost(tour, dist_mat, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j

    if best_move is not None:
        return best_delta, swap(tour, *best_move)
    return 0, tour


def relocate(tour, i, j):
    new_tour = tour.copy()
    n = new_tour.pop(i)
    new_tour.insert(j, n)
    return new_tour


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


def get_relocate_tour(tour, dist_mat, fixed_i=None):
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)

    if fixed_i is not None:
        i = fixed_i
        assert i > 0 and i < len(tour) - 1
        for j in idxs:
            if i == j:
                continue
            delta = relocate_cost(tour, dist_mat, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
    else:
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


def nearest_neighbor(adj_mat):
# Unused, for debugging and comparison only.
    tour = [0]
    while len(tour) < adj_mat.shape[0]:
        i = tour[-1]
        neighbours = []
        for j in range(adj_mat.shape[0]):
            if j != i and j not in tour:
                if i < j:
                    neighbours.append((j, adj_mat[i, j]))
                else:
                    neighbours.append((j, adj_mat[j, i]))
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(0)
    return tour


def local_search(init_tour, dist_mat):
    cur_tour = init_tour
    improved = True
    while improved:
        improved = False
        for operator in [get_two_opt_tour, get_relocate_tour]:
            delta, new_tour = operator(cur_tour, dist_mat)
            if delta < 0:
                improved = True
                cur_tour = new_tour
    return cur_tour


def tsp_ls(np_points, tours, adj_mat, device="cpu", **kwargs):
    '''
    Vanilla local search
	Output: tours, shape (parallel_sampling, N + 1)
	Reference: https://github.com/proroklab/gnngls
	'''
    solved_tours = []
    dist_mat = scipy.spatial.distance_matrix(np_points, np_points)
    if len(adj_mat.shape) == 2:
        adj_mat = np.expand_dims(adj_mat, 0)
        tours = [tours]
    if isinstance(tours, np.ndarray):
        tours = tours.tolist()
    for i in range(adj_mat.shape[0]):
        tours = tours[i]
        tours = local_search(tours, dist_mat)
        solved_tours.append(tours)
    return solved_tours    


def tsp_gls(np_points, tours, adj_mat, device="cpu", **kwargs):
    '''
    Regret guided local search
	Output: tours, shape (parallel_sampling, N + 1)
	Reference: https://github.com/proroklab/gnngls
	'''
    solved_tours = []
    eva = TSPEvaluator(np_points)
    num_nodes = np_points.shape[0]
    dist_mat = scipy.spatial.distance_matrix(np_points, np_points)
    adj_mat = -1 * adj_mat
    
    if isinstance(tours, np.ndarray):
        tours = tours.tolist()
    if len(adj_mat.shape) == 2:
        adj_mat = np.expand_dims(adj_mat, 0)
        tours = [tours]

    for i in range(adj_mat.shape[0]):
        adj_mat, tours = adj_mat[i], tours[i]
        adj_mat = np.where(adj_mat > 0, adj_mat, 0)
        init_cost = eva.evaluate(tours)
        k = 0.1 * init_cost / num_nodes
        cur_tour = local_search(tours, dist_mat)
        cur_cost = eva.evaluate(cur_tour)
        best_tour, best_cost = cur_tour, cur_cost
 
        iter_i = 0
        t_lim = time.time() + kwargs['time_limit']
        penalty_mat = np.zeros((num_nodes, num_nodes))
        while time.time() < t_lim:
            # perturbation
            moves = 0
            while moves < kwargs['perturbation_moves']:
                # penalize edge
                max_util = 0
                max_util_e = None
                for src, dst in zip(cur_tour[:-1], cur_tour[1:]):
                    if src > dst:
                        src, dst = dst, src
                    cur_regret = adj_mat[src, dst]
                    cur_pennalty = penalty_mat[src, dst]
                    util = cur_regret / (1 + cur_pennalty)
                    if util > max_util or max_util_e is None:
                        max_util = util
                        max_util_e = [src, dst]
                
                if max_util_e is None:
                    raise ValueError("max_util_e cannot be None")
                src, dst = max_util_e
                penalty_mat[src, dst] += 1.
                penalty_mat[dst, src] += 1.
                edge_weight_guided = dist_mat + k * penalty_mat

                # apply operator to edge
                for n in max_util_e:
                    if n != 0:  # not the depot
                        i = cur_tour.index(n)
                        for operator in [get_two_opt_tour, get_relocate_tour]:
                            moved = False
                            delta, new_tour = operator(cur_tour, edge_weight_guided, i)

                            if delta < 0:
                                cur_cost = eva.evaluate(new_tour)
                                cur_tour = new_tour
                                moved = True
                            moves += moved

            # optimisation
            cur_tour = local_search(cur_tour, dist_mat)
            cur_cost = eva.evaluate(cur_tour)
            if cur_cost < best_cost:
                best_tour, best_cost = cur_tour, cur_cost
            iter_i += 1
        solved_tours.append(best_tour)
    return solved_tours
