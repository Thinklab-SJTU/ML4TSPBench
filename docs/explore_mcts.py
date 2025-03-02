import os
import torch
import ctypes
import numpy as np
from search.c_mcts import c_mcts
from scipy.spatial.distance import cdist
from search.rg_mcts import random_greedy
from search.beam_search import Beamsearch, tour_nodes_to_tour_len, is_valid_tour
from models.utils.evaluator import TSPEvaluator
from tqdm import tqdm
from search.beam_search import Beamsearch 
import matplotlib.pyplot as plt
import torch.nn.functional as F


def generate_random_array(nodes_num: int):
    arr = np.arange(1, nodes_num)
    np.random.shuffle(arr)
    result = np.insert(arr, [0, len(arr)], [0, 0])
    return result


def get_heatmap(heatmap: np.ndarray, np_points: np.ndarray=None, heatmap_type: str="raw"):
    if heatmap_type == "raw":
        return heatmap
    elif heatmap_type == "ones":
        return np.ones_like(heatmap)
    elif heatmap_type == "distance":
        distance = cdist(np_points, np_points) + 1
        return F.softmax(torch.from_numpy(1 / distance), dim=-1).numpy()
    elif heatmap_type == "random":
        return abs(np.random.random(size=heatmap.shape))


def rmcts(
    tour: np.ndarray,
    nodes_num: int, 
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    mcts_max_depth: int=10,
    max_iterations: int=1000,
    heatmap_type: str="raw"
):
    """
    explore the relationship of the tour length before and after the mcts process
    """ 
    adj_mat = get_heatmap(adj_mat, np_points, heatmap_type)
    heatmap = adj_mat.reshape(-1)
    nodes_coords = np_points.reshape(-1)

    np_tour = tour.astype(np.int16)
    mcts_tour = c_mcts(
        np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        nodes_num,
        mcts_max_depth,
        max_iterations,
    )
    mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
    mcts_tour = mcts_tour.tolist()
    mcts_tour.append(mcts_tour[0])
             
    tsp_solver = TSPEvaluator(np_points)
    tour_cost = tsp_solver.evaluate(np_tour)
    mcts_tour_cost = tsp_solver.evaluate(mcts_tour)
    return [tour_cost, mcts_tour_cost]


def rmcts_main(
    num_nodes: int=100,
    samples: int = 100,
    heatmap_type: str="raw",
    use_history: bool=True,
    save: bool=True
):
    heatmap = np.load(f"explore_mcts/test_heatmap_{num_nodes}.npy", allow_pickle=True)
    points = np.load(f"explore_mcts/test_points_{num_nodes}.npy", allow_pickle=True)
    
    filename = f"explore_mcts/rmcts_{num_nodes}_{heatmap_type}.npy"
    if os.path.exists(filename) and use_history:
        costs_pair_list = np.load(filename, allow_pickle=True).tolist()
    else:
        costs_pair_list = []
    for _ in tqdm(range(samples), desc='Exploring'):
        tour = generate_random_array(num_nodes)
        costs_pair = rmcts(tour, num_nodes, heatmap, points, heatmap_type=heatmap_type)
        costs_pair_list.append(costs_pair)
    costs_pairs = np.array(costs_pair_list)
    if save:
        np.save(filename, costs_pairs)
        

def rg_mcts(
    nodes_num: int, 
    adj_mat: np.ndarray, 
    np_points: np.ndarray,
    random_weight: int=1, 
    mcts_max_depth: int=10,
    max_iterations: int=1000,
    heatmap_type: str="raw"
):
    """
    explore the relationship of the tour length before and after the mcts process
    """
    nodes_coords = np_points.reshape(-1)
    tour = random_greedy(adj_mat)
    tour = np.ctypeslib.as_array(tour, shape=(nodes_num,))
    tour = tour.tolist()
    tour.append(tour[0])
    np_tour = np.array(tour, dtype=np.int16)

    adj_mat = get_heatmap(adj_mat, np_points, heatmap_type)
    heatmap = adj_mat.reshape(-1)
    mcts_tour = c_mcts(
        np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
        nodes_num,
        mcts_max_depth,
        max_iterations,
    )
    mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
    mcts_tour = mcts_tour.tolist()
    mcts_tour.append(mcts_tour[0])
             
    tsp_solver = TSPEvaluator(np_points)
    tour_cost = tsp_solver.evaluate(np_tour)
    mcts_tour_cost = tsp_solver.evaluate(mcts_tour)
    return [tour_cost, mcts_tour_cost]


def rg_mcts_main(    
    num_nodes: int=100,
    samples: int= 100,
    heatmap_type: str="raw",
    use_history: bool=True,
    save: bool=True
):
    heatmap = np.load(f"explore_mcts/test_heatmap_{num_nodes}.npy", allow_pickle=True)
    points = np.load(f"explore_mcts/test_points_{num_nodes}.npy", allow_pickle=True)
    
    filename = f"explore_mcts/rg_mcts_{num_nodes}_{heatmap_type}.npy"
    if use_history and os.path.exists(filename):
        costs_pair_list = np.load(filename, allow_pickle=True).tolist()
    else:
        costs_pair_list = []
    for idx in tqdm(range(samples), desc='Exploring'):
        random_weight = idx * 0.01
        costs_pair = rg_mcts(num_nodes, heatmap, points, random_weight, heatmap_type=heatmap_type)
        costs_pair_list.append(costs_pair)
    costs_pairs = np.array(costs_pair_list)
    if save:
        np.save(filename, costs_pairs)


def beam_mcts(
    nodes_num: int, 
    adj_mat: np.ndarray, 
    np_points: np.ndarray,
    beam_size: int=10,
    probs_type: str="raw",
    beam_random_smart: bool=False,
    mcts_max_depth: int=10,
    max_iterations: int=1000,
    heatmap_type: str="raw"
):
    """
    explore the relationship of the tour length before and after the mcts process
    """
    adj_mat = np.expand_dims(adj_mat, axis=0)
    batch_size, num_nodes, _ = adj_mat.shape
    x_edges_values = np.expand_dims(cdist(np_points, np_points), axis=0)
    
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
    
    origin_tour_list = []
    mcts_tour_list = []
    # mcts
    adj_mat = adj_mat.cpu().detach().numpy()
    adj_mat = get_heatmap(adj_mat, np_points, heatmap_type)
    heatmap = adj_mat.reshape(-1)
    for pos in tqdm(range(beam_size), desc='Exploring'):
        ends = pos * torch.ones(1, 1).long() # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        tour = hyp_tours
        tour = tour[0].tolist()
        tour.append(0)         
        np_tour = np.array(tour, dtype=np.int16)
        nodes_coords = np_points.reshape(-1)
        mcts_tour = c_mcts(
            np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
            heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
            nodes_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
            nodes_num,
            mcts_max_depth,
            max_iterations
        )
        mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
        mcts_tour = mcts_tour.tolist()
        mcts_tour.append(mcts_tour[0])
        mcts_tour = np.array(mcts_tour)
        origin_tour_list.append(np_tour)
        mcts_tour_list.append(mcts_tour)
    
    tsp_solver = TSPEvaluator(np_points)
    pairs = list()
    for idx in range(len(origin_tour_list)):
        np_tour = origin_tour_list[idx]
        mcts_tour = mcts_tour_list[idx]
        tour_cost = tsp_solver.evaluate(np_tour)
        mcts_tour_cost = tsp_solver.evaluate(mcts_tour)
        pairs.append([tour_cost, mcts_tour_cost])
    pairs = np.array(pairs)
    return pairs


def beam_mcts_main(    
    num_nodes: int=100,
    samples: int= 100,
    heatmap_type: str="raw",
    use_history: bool=True,
    save: bool=True
):
    heatmap = np.load(f"explore_mcts/test_heatmap_{num_nodes}.npy", allow_pickle=True)
    points = np.load(f"explore_mcts/test_points_{num_nodes}.npy", allow_pickle=True)
    filename = f"explore_mcts/beam_mcts_{num_nodes}_{heatmap_type}.npy"
    costs_pairs = beam_mcts(num_nodes, heatmap, points, samples, heatmap_type=heatmap_type)
    if use_history and os.path.exists(filename):
        last_costs_pair = np.load(filename, allow_pickle=True)
        costs_pairs = np.concatenate([costs_pairs, last_costs_pair], axis=0)
    if save:
        np.save(filename, costs_pairs)


def plot_scatter(
    array_list: list, 
    color_list: list,
    name_list: list,
    size_list: list,
    filename: str="explore_mcts/output.png", 
    min_x: int=0,
    max_x: int=60,
    title: str="Scatter Plot"
):  
    for i in range(len(array_list)):
        x = array_list[i][:, 0]
        y = array_list[i][:, 1]
        plt.scatter(x, y, s=size_list[i], c=color_list[i], label=name_list[i])
    plt.legend()
    plt.xlabel('origin_cost')
    plt.ylabel('mcts_cost')
    plt.title(title)
    plt.xlim(min_x, max_x)
    plt.savefig(filename)
    plt.clf()
    

def main():
    heatmap_type_list = ['raw', 'ones', 'distance', 'random']
    num_nodes_list = [50, 100]
    samples = 100
    for num_nodes in num_nodes_list:
        for heatmap_type in heatmap_type_list:
            rmcts_main(num_nodes=num_nodes, heatmap_type=heatmap_type, samples=samples)
            beam_mcts_main(num_nodes=num_nodes, heatmap_type=heatmap_type, samples=samples) 
            rg_mcts_main(num_nodes=num_nodes, heatmap_type=heatmap_type, samples=samples)


def plot():
    method_list = ['rmcts', 'beam_mcts', 'rg_mcts']
    num_nodes_list = [50, 100]
    heatmap_type_list = ['raw', 'ones', 'distance', 'random']
    color_list = ["#FFD700", "#008000", "#800080",  "#FF69B4"]
    size_list = [5] * 4
    for method in method_list:
        for num_nodes in num_nodes_list:
            name_list = []
            array_list = []
            filename = f"explore_mcts/tsp{num_nodes}_{method}.png"
            title = f"TSP{num_nodes}_{method}"
            for heatmap_type in heatmap_type_list:
                array = np.load(f"explore_mcts/{method}_{num_nodes}_{heatmap_type}.npy", allow_pickle=True)
                array_list.append(array)
                name_list.append(heatmap_type)
            plot_scatter(array_list, color_list, name_list, size_list, filename, title=title)
            

if __name__ == "__main__":
    main()
    plot()
    