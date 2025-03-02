import time
from tqdm import tqdm
import ctypes
import numpy as np
import torch
from search.c_mcts import c_mcts
from search.c_mcts_solver import c_mcts_solver
from utils.evaluator import TSPEvaluator
import matplotlib.pyplot as plt
from search.beam_search import Beamsearch
from search.greedy_search import tsp_greedy
from search.mcts_smooth import smooth_heatmap, smooth_heatmap_v2


##########################################################
#                         RMCTS                          #
##########################################################

def generate_random_array(nodes_num: int):
    arr = np.arange(1, nodes_num)
    np.random.shuffle(arr)
    result = np.insert(arr, [0, len(arr)], [0, 0])
    return result


def rmcts_gap_time(
    heatmap: np.ndarray, 
    points: np.ndarray,
    gt_lengths_list: np.ndarray,
    limit_time: float=1.0, 
    record_samples: int=10,
    smooth: bool=False,
    smooth_v2: bool=False,
):
    nums = heatmap.shape[0]
    nodes_num = heatmap.shape[-1]
    lengths_list = np.zeros(shape=(nums, record_samples))
    time_list = np.zeros(shape=(nums, record_samples))
    time_slice = limit_time / record_samples
    for idx in tqdm(range(nums), desc='Running'):
        # Gain current heatmap and current points
        if smooth:
            heatmap[idx] = smooth_heatmap(heatmap[idx])
        elif smooth_v2:
            heatmap[idx] = smooth_heatmap_v2(heatmap[idx], points[idx])
        cur_heatmap = heatmap[idx].reshape(-1)
        cur_points = points[idx].reshape(-1)
        cur_solved_lengths = list()
        cur_tsp_solver = TSPEvaluator(points[idx])
        
        # Record the solved tour for each time slot
        for slice_num in range(record_samples):
            begin_time = time.time()
            while(time.time() - begin_time < time_slice):
                tour = generate_random_array(nodes_num)
                np_tour = np.array(tour, dtype=np.int16)
                mcts_tour = c_mcts(
                    np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
                    cur_heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    cur_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    nodes_num,
                    10,
                    0.005,
                    5000,
                )
                mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
                mcts_tour = mcts_tour.tolist()
                mcts_tour.append(mcts_tour[0])
                length = cur_tsp_solver.evaluate(mcts_tour)
                cur_solved_lengths.append(length)
            if slice_num == 0:
                time_list[idx][slice_num] = time.time() - begin_time
            else:
                time_list[idx][slice_num] = time.time() - begin_time + time_list[idx][slice_num-1]
            lengths_list[idx][slice_num] = np.min(np.array(cur_solved_lengths)).item()
    
    # gap_list and time_list processing
    gap_list = (lengths_list - gt_lengths_list) / gt_lengths_list * 100
    avg_gap_list = np.average(gap_list, axis=0)
    avg_time_list = np.average(time_list, axis=0)
    return avg_gap_list, avg_time_list


##########################################################
#                        RG MCTS                         #
##########################################################

def random_greedy(adj_mat: np.ndarray, random_weight: float=3.0):
    weight = np.exp((adj_mat + adj_mat.T) / 2 * random_weight)
    num_nodes = adj_mat.shape[0]
    np_tour = np.zeros(num_nodes + 1)
    mask = np.ones(shape=(num_nodes, num_nodes))
    mask[:, 0] = 0
    np.fill_diagonal(mask, 0)
    num = 0
    idx = 0
    while(num < num_nodes - 1):
        weight = weight * mask
        probs = weight[idx] / np.sum(weight[idx])
        seleted_idx = np.random.choice(np.arange(len(probs)), p=probs)
        num = num + 1
        np_tour[num] = seleted_idx
        idx = seleted_idx
        mask[:, seleted_idx] = 0
    np_tour = np_tour.astype(np.int16)
    return np_tour


def rg_mcts_gap_time(
    heatmap: np.ndarray, 
    points: np.ndarray,
    gt_lengths_list: np.ndarray,
    limit_time: float=1.0, 
    record_samples: int=10,
    smooth: bool=False,
    smooth_v2: bool=False
):
    nums = heatmap.shape[0]
    nodes_num = heatmap.shape[-1]
    lengths_list = np.zeros(shape=(nums, record_samples))
    time_list = np.zeros(shape=(nums, record_samples))
    time_slice = limit_time / record_samples
    for idx in tqdm(range(nums), desc='Running'):
        # Gain current heatmap and current points
        if smooth:
            heatmap[idx] = smooth_heatmap(heatmap[idx])
        elif smooth_v2:
            heatmap[idx] = smooth_heatmap_v2(heatmap[idx], points[idx])
        cur_heatmap = heatmap[idx].reshape(-1)
        cur_points = points[idx].reshape(-1)
        cur_solved_lengths = list()
        cur_tsp_solver = TSPEvaluator(points[idx])
        
        # Record the solved tour for each time slot
        for slice_num in range(record_samples):
            begin_time = time.time()
            while(time.time() - begin_time < time_slice):
                np_tour = random_greedy(heatmap[idx])
                mcts_tour = c_mcts(
                    np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
                    cur_heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    cur_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    nodes_num,
                    10,
                    0.005,
                    5000,
                )
                mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
                mcts_tour = mcts_tour.tolist()
                mcts_tour.append(mcts_tour[0])
                length = cur_tsp_solver.evaluate(mcts_tour)
                cur_solved_lengths.append(length)
            if slice_num == 0:
                time_list[idx][slice_num] = time.time() - begin_time
            else:
                time_list[idx][slice_num] = time.time() - begin_time + time_list[idx][slice_num-1]
            lengths_list[idx][slice_num] = np.min(np.array(cur_solved_lengths)).item()
    
    # gap_list and time_list processing
    gap_list = (lengths_list - gt_lengths_list) / gt_lengths_list * 100
    avg_gap_list = np.average(gap_list, axis=0)
    avg_time_list = np.average(time_list, axis=0)
    return avg_gap_list, avg_time_list


##########################################################
#                       BEAM MCTS                        #
##########################################################

def beam_mcts_gap_time(
    heatmap: np.ndarray, 
    points: np.ndarray,
    gt_lengths_list: np.ndarray,
    limit_time: float=1.0, 
    record_samples: int=10,
    smooth: bool=False,
    smooth_v2: bool=False
):
    heatmap += 1e-14
    nums = heatmap.shape[0]
    nodes_num = heatmap.shape[-1]
    lengths_list = np.zeros(shape=(nums, record_samples))
    time_list = np.zeros(shape=(nums, record_samples))
    time_slice = limit_time / record_samples
    for idx in tqdm(range(nums), desc='Running'):
        # Gain current heatmap and current points
        if smooth:
            heatmap[idx] = smooth_heatmap(heatmap[idx])
        elif smooth_v2:
            heatmap[idx] = smooth_heatmap_v2(heatmap[idx], points[idx])
        cur_heatmap = heatmap[idx].reshape(-1)
        cur_points = points[idx].reshape(-1)
        cur_solved_lengths = list()
        cur_tsp_solver = TSPEvaluator(points[idx])
        
        # beam search
        beam_begin_time = time.time()
        adj_mat = torch.from_numpy(heatmap[idx]).unsqueeze(dim=0)
        nodes_num = adj_mat.shape[-1]

        beamsearch = Beamsearch(
            beam_size=record_samples, 
            batch_size=1, 
            num_nodes=nodes_num,
            probs_type="logits" 
        )
        beam_time = time.time() - beam_begin_time
        part_beam_time = beam_time / record_samples
        
        beam_mat = torch.log(torch.clamp(adj_mat, min=1e-14, max=0.9999)) - 1e-10
        trans_probs = beam_mat.gather(1, beamsearch.get_current_state())
        for _ in range(nodes_num - 1):
            beamsearch.advance(trans_probs)
            trans_probs = beam_mat.gather(1, beamsearch.get_current_state())
        
        # Record the solved tour for each time slot
        cur_beam_idx = 0
        for slice_num in range(record_samples):
            begin_time = time.time()
            while(time.time() - begin_time < time_slice):
                ends = cur_beam_idx * torch.ones(1, 1).long()
                cur_beam_idx += 1
                hyp_tours = beamsearch.get_hypothesis(ends)
                np_tour = hyp_tours[0].cpu().numpy().astype(np.int16)
                mcts_tour = c_mcts(
                    np_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
                    cur_heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    cur_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                    nodes_num,
                    10,
                    0.01,
                    5000,
                )
                mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
                mcts_tour = mcts_tour.tolist()
                mcts_tour.append(mcts_tour[0])
                length = cur_tsp_solver.evaluate(mcts_tour)
                cur_solved_lengths.append(length)
            if slice_num == 0:
                time_list[idx][slice_num] = time.time() - begin_time + part_beam_time
            else:
                time_list[idx][slice_num] = time.time() - begin_time + \
                    time_list[idx][slice_num-1] + part_beam_time
            lengths_list[idx][slice_num] = np.min(np.array(cur_solved_lengths)).item()

    # gap_list and time_list processing
    gap_list = (lengths_list - gt_lengths_list) / gt_lengths_list * 100
    avg_gap_list = np.average(gap_list, axis=0)
    avg_time_list = np.average(time_list, axis=0)
    return avg_gap_list, avg_time_list


##########################################################
#                        TWO OPT                         #
##########################################################

def two_opt_gap_time(
    heatmap: np.ndarray, 
    points: np.ndarray,
    gt_lengths_list: np.ndarray,
    limit_time: float=1.0, 
    record_samples: int=10
):
    points = points.astype("float64")
    nums = heatmap.shape[0]
    lengths_list = np.zeros(shape=(nums, record_samples))
    time_list = np.zeros(shape=(nums, record_samples))
    time_slice = limit_time / record_samples
    
    for idx in tqdm(range(nums), desc='Running'):
        # Gain current heatmap and current points
        cur_tsp_solver = TSPEvaluator(points[idx])
        
        # greedy search
        greedy_begin_time = time.time()
        np_tour = tsp_greedy(np.expand_dims(heatmap[idx], axis=0), points[idx])
        np_tour = np.array(np_tour).astype("int64")
        greedy_time = time.time() - greedy_begin_time
        part_greedy_time = greedy_time / record_samples
        
        # 2opt           
        with torch.inference_mode():
            cuda_points = torch.from_numpy(points[idx]).to('cuda')
            cuda_tour = torch.from_numpy(np_tour.copy()).to('cuda')
            batch_size = cuda_tour.shape[0]
            min_change = -1.0
    
            for slice_num in range(record_samples):
                begin_time = time.time()
                while(time.time() - begin_time < time_slice and min_change < 0.0):
                    points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
                    points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
                    points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
                    points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

                    A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
                    A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
                    A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
                    A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

                    change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
                    valid_change = torch.triu(change, diagonal=2)

                    min_change = torch.min(valid_change)
                    flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
                    min_i = torch.div(flatten_argmin_index, len(points[idx]), rounding_mode='floor')
                    min_j = torch.remainder(flatten_argmin_index, len(points[idx]))

                    if min_change < -1e-6:
                        for i in range(batch_size):
                            cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                    else:
                        break
            
                if slice_num == 0:
                    time_list[idx][slice_num] = time.time() - begin_time + part_greedy_time
                else:
                    time_list[idx][slice_num] = time.time() - begin_time + \
                        time_list[idx][slice_num-1] + part_greedy_time
                lengths_list[idx][slice_num] = cur_tsp_solver.evaluate(cuda_tour[0])
        
        
    # gap_list and time_list processing
    gap_list = (lengths_list - gt_lengths_list) / gt_lengths_list * 100
    avg_gap_list = np.average(gap_list, axis=0)
    avg_time_list = np.average(time_list, axis=0)
    return avg_gap_list, avg_time_list    


##########################################################
#                      MCTS Solver                       #
##########################################################

def mcts_solver_gap_time(
    heatmap: np.ndarray, 
    points: np.ndarray,
    gt_lengths_list: np.ndarray,
    limit_time: float=1.0, 
    record_samples: int=10,
    smooth: bool=False,
    smooth_v2: bool=False
):
    nums = heatmap.shape[0]
    nodes_num = heatmap.shape[-1]
    lengths_list = np.zeros(shape=(nums, record_samples))
    time_list = np.zeros(shape=(nums, record_samples))
    time_slice = limit_time / record_samples
    for idx in tqdm(range(nums), desc='Running'):
        # Gain current heatmap and current points
        if smooth:
            heatmap[idx] = smooth_heatmap(heatmap[idx])
        elif smooth_v2:
            heatmap[idx] = smooth_heatmap_v2(heatmap[idx], points[idx])
        cur_heatmap = heatmap[idx].reshape(-1)
        cur_points = points[idx].reshape(-1)
        cur_tsp_solver = TSPEvaluator(points[idx])
        
        # Record the solved tour for each time slot
        for slice_num in range(record_samples):
            mcts_param_t = (slice_num + 1) * time_slice
            begin_time = time.time()
            mcts_tour = c_mcts_solver(
                cur_heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                cur_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
                nodes_num,
                10,
                ctypes.c_float(mcts_param_t),
                5000,
            )
            mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
            mcts_tour = mcts_tour.tolist()
            mcts_tour.append(mcts_tour[0])
            length = cur_tsp_solver.evaluate(mcts_tour)
            time_list[idx][slice_num] = time.time() - begin_time
            lengths_list[idx][slice_num] = length
    
    # gap_list and time_list processing
    gap_list = (lengths_list - gt_lengths_list) / gt_lengths_list * 100
    avg_gap_list = np.average(gap_list, axis=0)
    avg_time_list = np.average(time_list, axis=0)
    return avg_gap_list, avg_time_list


##########################################################
#                          PLOT                          #
##########################################################

def plot(gaps: np.ndarray, times: np.ndarray, save_path: str):
    plt.plot(times, gaps)
    plt.xlabel('Times')
    plt.ylabel('Gaps')
    plt.title('Plot of Gaps over Times')
    plt.savefig(save_path)
    

##########################################################
#                          MAIN                          #
##########################################################

def tsp500_main(method_list: list(), limit_time: float=0.01, record_samples: int=100):
    for method in method_list:
        # load the data
        tsp500_heatmap = np.load(f"gap_time/tsp500/tsp500_{method}_heatmap.npy", allow_pickle=True)
        tsp500_points = np.load("gap_time/tsp500/tsp500_points.npy", allow_pickle=True)
        tsp500_gt_tours = np.load("gap_time/tsp500/tsp500_gt_tours.npy", allow_pickle=True)

        # gain the gt_length
        gt_lengths_list = np.zeros(tsp500_heatmap.shape[0])
        for idx in range(tsp500_heatmap.shape[0]):
            eva = TSPEvaluator(points=tsp500_points[idx])
            gt_lengths_list[idx] = eva.evaluate(tsp500_gt_tours[idx])
        gt_lengths_list = np.repeat(gt_lengths_list.reshape(-1 , 1), record_samples, axis=1)
        
        # rmcts
        rmcts_result, rmcts_time_list = rmcts_gap_time(
            heatmap=tsp500_heatmap, 
            points=tsp500_points,
            gt_lengths_list=gt_lengths_list,
            limit_time=limit_time, 
            record_samples=record_samples,
            smooth_v2=True
        )
        print(f"tsp500_{method} rmcts gap:", rmcts_result)
        print(f"tsp500_{method} rmcts time:", rmcts_time_list)
    
        # rg_mcts
        rg_mcts_result, rg_mcts_time_list = rg_mcts_gap_time(
            heatmap=tsp500_heatmap, 
            points=tsp500_points,
            gt_lengths_list=gt_lengths_list,
            limit_time=limit_time, 
            record_samples=record_samples,
            smooth_v2=True
        )
        print(f"tsp500_{method} rg_mcts gap:", rg_mcts_result)
        print(f"tsp500_{method} rg_mcts time:", rg_mcts_time_list)
        
        # beam_mcts
        beam_mcts_result, beam_mcts_time_list = beam_mcts_gap_time(
            heatmap=tsp500_heatmap, 
            points=tsp500_points,
            gt_lengths_list=gt_lengths_list,
            limit_time=limit_time, 
            record_samples=record_samples,
            smooth_v2=True
        )
        print(f"tsp500_{method} beam_mcts gap:", beam_mcts_result)
        print(f"tsp500_{method} beam_mcts time:", beam_mcts_time_list)

        # 2opt
        two_opt_result, two_opt_time_list = two_opt_gap_time(
            heatmap=tsp500_heatmap, 
            points=tsp500_points,
            gt_lengths_list=gt_lengths_list,
            limit_time=limit_time,
            record_samples=record_samples
        )
        print(f"tsp500_{method} two_opt gap:", two_opt_result)
        print(f"tsp500_{method} two_opt time:", two_opt_time_list)

        # mcts_solver
        mcts_solver_result, mcts_solver_time_list = mcts_solver_gap_time(
            heatmap=tsp500_heatmap, 
            points=tsp500_points,
            gt_lengths_list=gt_lengths_list,
            limit_time=limit_time,
            record_samples=record_samples
        )
        print(f"tsp500_{method} two_opt gap:", mcts_solver_result)
        print(f"tsp500_{method} two_opt time:", mcts_solver_time_list)
        
if __name__ == "__main__":
    # method_list = ['gnn', 'gnn_wise', 'diffusion']
    method_list = ['diffusion']
    tsp500_main(method_list, limit_time=0.04, record_samples=2)