import os
import shutil
import time
from utils.args import generate_arg_parser
import numpy as np
from utils.utils import tab_printer, get_data_from_tsp_file, get_tour_from_tour_file
from solvers.pyconcorde import TSPSolver
import tsplib95
import lkh
from utils.evaluator import TSPEvaluator
import itertools
import warnings
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


#########################################
#          Generate Nodes_coord         #
#########################################

def generate_uniform(batch_size: int, num_nodes: int):
    return np.random.random([batch_size, num_nodes, 2])


def generate_cluster(
    batch_size: int,
    num_nodes: int, 
    num_clusters: int=10, 
    cluster_std: float=0.1
):
    nodes_coords = np.zeros([batch_size, num_nodes, 2])
    for i in range(batch_size):
        cluster_centers = np.random.random([num_clusters, 2]) 
        cluster_points = []
    for center in cluster_centers:
        points = np.random.normal(
            loc=center, 
            scale=cluster_std, 
            size=(num_nodes // num_clusters, 2)
        )
        cluster_points.append(points)
    nodes_coords[i] = np.concatenate(cluster_points, axis=0)
    return nodes_coords


def generate_cluster_fixed_centers(batch_size: int, num_nodes: int):
    assert num_nodes in [100, 500]
    nodes_coords = np.zeros([batch_size, num_nodes, 2])
    for i in range(batch_size):
        num_clusters_axis = 2 if num_nodes == 100 else 5
        num_clusters = num_clusters_axis ** 2
        cluster_centers_axis = np.linspace(0, 1, num_clusters_axis * 2 + 1)[1::2]
        x, y = np.meshgrid(cluster_centers_axis, cluster_centers_axis)
        cluster_centers = [[x, y] for x, y in zip(x.flatten(), y.flatten())]
        scale = 1 / (num_clusters_axis * 3 * 3) if num_nodes == 100 else 1 / (num_clusters_axis * 3 * 3)
        cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(loc=center, scale=scale, size=(num_nodes // num_clusters, 2))
            cluster_points.append(points)
        nodes_coords[i] = np.concatenate(cluster_points, axis=0)
    return nodes_coords


def generate_gaussian(
    batch_size: int, 
    num_nodes: int, 
    mean_x: float=0.0, 
    mean_y: float=0.0, 
    std: float=1.0
):
    return np.random.normal(
        loc=[mean_x, mean_y], 
        scale=std, 
        size=(batch_size, num_nodes, 2)
    )


def generate_nodes_coord(batch_size: int, num_nodes: int, opts):
    if opts.type == "uniform":
        return generate_uniform(batch_size, num_nodes)
    elif opts.type == "cluster":
        return generate_cluster(
            batch_size=batch_size, 
            num_nodes=num_nodes, 
            num_clusters=opts.num_clusters,
            cluster_std=opts.cluster_std
        )
    elif opts.type == "cluster_fixed_centers":
        return generate_cluster_fixed_centers(batch_size, num_nodes)
    elif opts.type == "gaussian":
        return generate_gaussian(
            batch_size=batch_size, 
            num_nodes=num_nodes, 
            mean_x=opts.mean_x,
            mean_y=opts.mean_y,
            std=opts.gaussian_std
        )
        
        
#########################################
#           Traditional Solver          #
#########################################

def solve_tsp(solver, nodes_coord, max_trials=100):
    num_nodes = nodes_coord.shape[0]
    if solver == "concorde":
        scale = 1e6
        solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="GEO")
        solution = solver.solve(verbose=False)
        tour = solution.tour
    elif solver == "lkh":
        scale = 1e6
        lkh_path = 'LKH'
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = num_nodes
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {n + 1: nodes_coord[n] * scale for n in range(num_nodes)}
        solution = lkh.solve(lkh_path, problem=problem, max_trials=max_trials, runs=10)
        tour = [n - 1 for n in solution[0]]
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return tour


#########################################
#            Generate files             #
#########################################

def generate(opts):
    with open(opts.filename, "w") as f:
        start_time = time.time()
        cnt = 0
        for b_idx in range(opts.num_samples // opts.batch_size):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes+1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            # batch_nodes_coord = np.random.random([opts.batch_size, num_nodes, 2])
            batch_nodes_coord = generate_nodes_coord(opts.batch_size, num_nodes, opts)

            solve_tsp_with_opts = partial(solve_tsp, opts.solver)
            with Pool(opts.batch_size) as p:
                tours = p.map(solve_tsp_with_opts, [batch_nodes_coord[idx] for idx in range(opts.batch_size)], opts.max_trials)

            for idx, tour in enumerate(tours):
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(str(" ") + str('output') + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")
                    if opts.calc_regret:
                        if not os.path.exists(opts.regret_dir):
                            os.makedirs(opts.regret_dir)
                        opt_tour = list(tour) + [0]
                        reg_mat = calc_regret(batch_nodes_coord[idx], opt_tour)
                        np.save(os.path.join(opts.regret_dir, f'{cnt}.npy'), reg_mat)
                    cnt += 1

        end_time = time.time() - start_time
        assert b_idx == opts.num_samples // opts.batch_size - 1
        f.close()
    
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time/60:.1f}m")
    print(f"Average time: {end_time/opts.num_samples:.1f}s")
    
    
#########################################
#                TSPLIBS                #
#########################################

def read_tsplibs(tsp_problem, tsp_opt_tour):
    problem_name = os.path.basename(os.path.splitext(tsp_problem)[0])
    tour_name = os.path.basename(os.path.splitext(tsp_opt_tour)[0])
    assert problem_name + '.opt' == tour_name
    try:
        nodes_coord = get_data_from_tsp_file(tsp_problem).node_coords
    except ValueError:
        return
    if nodes_coord is None:
        return
    nodes_coord = nodes_coord.squeeze()
    tour = get_tour_from_tour_file(tsp_opt_tour)
    save_dir = os.path.dirname(tsp_problem).replace('raw/read', 'processed')
    filename = os.path.join(save_dir, 'tsp_tsplibs_' + problem_name + '.txt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename, "w") as f:
        f.write(" ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord))
        f.write(str(" ") + str('output') + str(" "))
        f.write(str(" ").join( str(node_idx+1) for node_idx in tour))
        f.write(str(" ") + str(tour[0]+1) + str(" "))
        f.write("\n")


def solve_tsplibs(filename):
    try:
        nodes_coord = get_data_from_tsp_file(filename).node_coords
    except ValueError:
        return 
    nodes_coord = nodes_coord.squeeze()
    save_dir = os.path.dirname(filename).replace('raw/calculate', 'processed')
    save_path = os.path.join(save_dir, 'tsp_tsplibs_' + os.path.basename(os.path.splitext(filename)[0]) + '.txt')   
    solver = TSPSolver.from_tspfile(filename)
    solution = solver.solve()
    with open(save_path, "w") as f:
        f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
        f.write( str(" ") + str('output') + str(" ") )
        f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour) )
        f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
        f.write( "\n" )
        

#########################################
#                 Divide                #
#########################################

def divide_file(filename, train_filename, valid_filename, test_filename, \
                train_ratio, val_ratio, test_ratio, regret_dir=None):
    with open(filename, "r") as f:
        data = f.readlines()   
        
    total_samples = len(data)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    train_data = data[: train_samples]
    val_data = data[train_samples: train_samples + val_samples]
    test_data = data[train_samples + val_samples:]

    with open(train_filename, 'w') as file:
        file.writelines(train_data)
    with open(valid_filename, 'w') as file:
        file.writelines(val_data)
    with open(test_filename, 'w') as file:
        file.writelines(test_data)     

    if regret_dir:
        assert os.path.exists(regret_dir)
        for root, dir, file in os.walk(regret_dir):
            file.sort(key=lambda x: int(x.split('.')[0]))
            for i, reg_file in enumerate(file):
                if i < train_samples:
                    shutil.move(os.path.join(root, reg_file), 
                                os.path.join(regret_dir, f'train_{i}.npy'))
                elif i < train_samples + val_samples:
                    shutil.move(os.path.join(root, reg_file), 
                                os.path.join(regret_dir, f'val_{i - train_samples}.npy'))
                else:
                    shutil.move(os.path.join(root, reg_file), 
                                os.path.join(regret_dir, f'test_{i - train_samples - val_samples}.npy'))
            break
        
        
#########################################
#                Resolve                #
#########################################

def resolve(filename, target_file=None, solver='lkh', batch_size=16, max_trials=100):
    warnings.filterwarnings('ignore')
    with open(filename, 'r') as file:
        nodes_coords = list() 
        for line in file:
            line = line.strip()
            points = line.split(' output ')[0]
            points = points.split(' ')
            points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
            nodes_coords.append(points)
        
    num_samples = len(nodes_coords)    
    nodes_coords = np.array(nodes_coords)
    num_nodes = nodes_coords.shape[1]
    nodes_coords = nodes_coords.reshape(-1, batch_size, num_nodes, 2)

    if target_file is None:
        target_file = filename[:-4] + "_" + solver +  "_" + str(max_trials) +".txt"

    with open(target_file, "w") as f:
        start_time = time.time()
        cnt = 0
        for b_idx in range(num_samples // batch_size):
            batch_nodes_coord = nodes_coords[b_idx]
            tours = []
            for idx in range(batch_size):
                tour = solve_tsp(solver, batch_nodes_coord[idx], max_trials=max_trials)
                tours.append(tour)

            for idx, tour in enumerate(tours):
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(str(" ") + str('output') + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")
                    cnt += 1

        end_time = time.time() - start_time
        assert b_idx == num_samples // batch_size - 1
        f.close()
    
    print(f"Completed Resolution of {filename}.")
    print(f"Total time: {end_time/60:.1f}m")
    print(f"Average time: {end_time/num_samples:.1f}s")
    
    
###################################################
#       Util functions for computing regret       #
###################################################

def fixed_edge_tour(points, e, scale=1e6, lkh_path='LKH', **kwargs):
    warnings.filterwarnings('ignore')
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = points.shape[0]
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {n + 1: scale * points[n] for n in range(points.shape[0])}
    problem.fixed_edges = [[n + 1 for n in e]]
    solution = lkh.solve(lkh_path, problem=problem, **kwargs)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def gen_regret(points, eva, opt_tour, edge):
    i, j = edge
    tour = fixed_edge_tour(points, (i, j), max_trials=10, runs=10)
    cost = eva.evaluate(tour)
    opt_cost = eva.evaluate(opt_tour)
    regret = (cost - opt_cost) / opt_cost
    return i, j, regret


def calc_regret(points, opt_tour):
    num_nodes = points.shape[0]
    reg_mat = np.zeros((num_nodes, num_nodes))
    eva = TSPEvaluator(points)
    for i, j in itertools.combinations(range(num_nodes), 2):
        tour = fixed_edge_tour(points, (i, j), max_trials=100, runs=10)
        cost = eva.evaluate(tour)
        opt_cost = eva.evaluate(opt_tour)
        regret = (cost - opt_cost) / opt_cost
        reg_mat[i, j] = reg_mat[j, i] = regret
    return reg_mat


def read_txt(line):
    line = line.strip()
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1
    return points, tour


def calc_regret_from_txt(opts):
    cnt = 0
    samples = []
    with open(opts.filename, 'r') as f:
        lines = f.read().splitlines()
        print(len(lines))
    # with Pool(opts.batch_size) as p:
    #   samples = p.map(read_txt, lines)
        for line in tqdm(lines, desc='reading lines'):
            line = line.strip()
            points = line.split(' output ')[0]
            points = points.split(' ')
            points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
            tour = line.split(' output ')[1]
            tour = tour.split(' ')
            tour = np.array([int(t) for t in tour])
            tour -= 1
            samples.append((points, tour))
        for batch_idx in tqdm(range(len(lines) // opts.batch_size)):
            with Pool(opts.batch_size) as p:
                reg_mats = p.map(calc_regret, samples[batch_idx * opts.batch_size: (batch_idx + 1) * opts.batch_size])
            for reg_mat in tqdm(reg_mats, desc='saving regret'):
                np.save(os.path.join(opts.regret_dir, f'{cnt}.npy'), reg_mat)
                cnt += 1
                
                
#########################################
#                  Main                 #
#########################################

if __name__ == "__main__":
    opts = generate_arg_parser()

    opts.filename = opts.filename if opts.filename else \
        f"data/{opts.type}/tsp_{opts.type}_{opts.min_nodes}-{opts.max_nodes}_{opts.solver}.txt"
    opts.train_filename = opts.train_filename if opts.train_filename else \
        f"data/{opts.type}/tsp_{opts.type}_{opts.min_nodes}-{opts.max_nodes}_train.txt"
    opts.valid_filename = opts.valid_filename if opts.valid_filename else \
        f"data/{opts.type}/tsp_{opts.type}_{opts.min_nodes}-{opts.max_nodes}_val.txt"
    opts.test_filename = opts.test_filename if opts.test_filename \
        else f"data/{opts.type}/tsp_{opts.type}_{opts.min_nodes}-{opts.max_nodes}_test.txt"
    if opts.calc_regret:
        opts.regret_dir = opts.regret_dir if opts.regret_dir else \
        f"data/{opts.type}/tsp_{opts.type}_{opts.min_nodes}-{opts.max_nodes}_regret"

    ratios = opts.ratio.split(":")
    total = sum(map(int, ratios))
    train_ratio, val_ratio, test_ratio = (int(ratio) / total for ratio in ratios)
    
    if opts.type == 'tsplibs':
        if opts.mode == 'read':
            filenames = np.array(sorted(os.listdir(opts.tsplibs_path))).reshape(-1,2)
            for tsplib_data in filenames:
                tsp_problem = os.path.join(opts.tsplibs_path, tsplib_data[1])
                tsp_opt_tour = os.path.join(opts.tsplibs_path, tsplib_data[0])
                read_tsplibs(tsp_problem, tsp_opt_tour)
        else:
            filenames = np.array(os.listdir(opts.tsplibs_path))
            for filename in filenames:
                solve_tsplibs(os.path.join(opts.tsplibs_path, filename))
  
    elif opts.type == 'txt':
        if not os.path.exists(opts.regret_dir):
            os.makedirs(opts.regret_dir)
        calc_regret_from_txt(opts)
        divide_file(opts.filename, opts.train_filename, opts.valid_filename, \
            opts.test_filename, train_ratio, val_ratio, test_ratio, opts.regret_dir)
  
    else:
        assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
        np.random.seed(opts.seed)
        tab_printer(opts)
        generate(opts)
        divide_file(opts.filename, opts.train_filename, opts.valid_filename, \
            opts.test_filename, train_ratio, val_ratio, test_ratio, opts.regret_dir)
    