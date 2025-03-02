import numpy as np
from texttable import Texttable
from scipy.spatial.distance import cdist
import os
import tsplib95
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
import pickle
from typing import Union


############################################
##               TSP_DATA                 ##
############################################

class TSP_DATA:
    """
    Represents a batch of TSP problems, each with a fixed number of nodes and random node node_coords.

    Attributes:
        node_coords (ndarray): A 3D numpy array of shape (batch, nodes_num, 2), 
            representing the node node_coords of eachTSP problem in the batch.
        edge_weights (ndarray): A 3D numpy array of shape (batch, nodes_num, nodes_num), 
            representing the distance matrix of each TSP problem in the batch.
        tour (ndarray): A 2D numpy array of shape (batch, nodes_num), representing the tour of
            each TSP problem in the batch, if available.
            
    """
    def __init__(self, node_coords: np.ndarray=None, edge_weights: np.ndarray=None):
        if node_coords is None:
            self.node_coords = None
            if edge_weights is None:
                raise ValueError("node_coords and edge_weights cannot be both None")
            self.edge_weights = edge_weights
        else:
            if node_coords.ndim == 2:
                node_coords = np.expand_dims(node_coords,axis=0)
            assert node_coords.ndim == 3
            self.node_coords = node_coords
            self.edge_weights = np.array([cdist(coords,coords) for coords in self.node_coords])
        self.tour = None
    
    def __repr__(self):
        if self.node_coords is None:
            self.message = "edge_weights = {}".format(self.edge_weights.shape)
        else:
            self.message = "node_coords = {}, edge_weights = {}".format(self.node_coords.shape,self.edge_weights.shape)
        return f"{self.__class__.__name__}({self.message})"


class TSPGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sparse_factor=-1, regret_path=None, scale_regret=True, mode=None):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.regret_path = regret_path
        self.file_lines = open(data_file).read().splitlines()
        self.reg_scaler = None
        if self.regret_path is not None and scale_regret:
            self.n_nodes = int(data_file.split('/')[-1].split('_')[0][3:])
            self.reg_scaler = pickle.load(open(os.path.join(self.regret_path, f'tsp{self.n_nodes}_regret_scaler.pkl'), 'rb'))
            self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes), diagonal=1).bool()
        if mode is not None:
            self.mode = mode
        else:
            if 'train' in self.data_file:
                self.mode = 'train'
            elif 'val' in self.data_file:
                self.mode = 'val'
            else:
                self.mode = 'test'

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()
        # Extract points
        points = line.split(' output ')[0]
        points = points.split(' ')
        points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
        # Extract tour
        tour = line.split(' output ')[1]
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        tour -= 1

        return points, tour

    def __getitem__(self, idx):
        points, tour = self.get_example(idx)
        if self.sparse_factor <= 0:
            # Return a densely connected graph
            adj_matrix = np.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
                adj_matrix[tour[i + 1], tour[i]] = 1
                
            if self.regret_path is not None and self.mode in ['train', 'val']:
                reg_idx = idx // 8 if self.mode == 'train' else idx
                reg_file = os.path.join(self.regret_path, f'{self.mode}_{reg_idx}.npy')
                reg_matrix = torch.from_numpy(np.load(reg_file))
                if self.reg_scaler is not None:
                    edge_reg = torch.masked_select(reg_matrix, self.mask).view(-1, 1) # shape: [# edges, 1]
                    reg_transformed = self.reg_scaler.transform(edge_reg.numpy())
                    reg_matrix[self.mask] = torch.from_numpy(reg_transformed.reshape(-1))
                return (
                    reg_matrix.float(),
                    torch.LongTensor(np.array([idx], dtype=np.int64)),
                    torch.from_numpy(points).float(),
                    torch.from_numpy(adj_matrix).float(),
                    torch.from_numpy(tour).long()
                )
            else:
                return (
                    torch.LongTensor(np.array([idx], dtype=np.int64)),
                    torch.from_numpy(points).float(),
                    torch.from_numpy(adj_matrix).float(),
                    torch.from_numpy(tour).long(),
                )
        else:
            # Return a sparse graph where each node is connected to its k nearest neighbors
            # k = self.sparse_factor
            sparse_factor = self.sparse_factor
            kdt = KDTree(points, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

            edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

            tour_edges = np.zeros(points.shape[0], dtype=np.int64)
            tour_edges[tour[:-1]] = tour[1:]
            tour_edges = torch.from_numpy(tour_edges)
            tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
            
            tour_edges_rv = np.zeros(points.shape[0], dtype=np.int64)
            tour_edges_rv[tour[1:]] = tour[0:-1]
            tour_edges_rv = torch.from_numpy(tour_edges_rv)
            tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)   

            graph_data = GraphData(x=torch.from_numpy(points).float(),
                                    edge_index=edge_index,
                                    edge_attr=tour_edges + tour_edges_rv)

            point_indicator = np.array([points.shape[0]], dtype=np.int64)
            edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)), # [N, 1]
                graph_data,
                torch.from_numpy(point_indicator).long(), # [B, N, 2]
                torch.from_numpy(edge_indicator).long(), # [B, N, N]
                torch.from_numpy(tour).long(), # [B, N+1]
                torch.from_numpy(dis_knn).reshape(-1)
            )


#############################################
##               Points Sparse             ##
#############################################

def sparse_points(
    points: Union[np.ndarray, torch.Tensor],
    sparse_factor: int,
    device: str="cpu"
):
    if type(points) == torch.Tensor:
        points = points.detach().cpu().numpy()
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    
    edge_index = list()
    for idx in range(points.shape[0]):
        kdt = KDTree(points[idx], leaf_size=30, metric='euclidean')
        _, idx_knn = kdt.query(points[idx], k=sparse_factor, return_distance=True)
        _edge_index_0 = torch.arange(points[idx].shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        _edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
        _edge_index = torch.stack([_edge_index_0, _edge_index_1], dim=0)
        edge_index.append(_edge_index.unsqueeze(dim=0))
    edge_index = torch.cat(edge_index, dim=0).to(device)

    points = torch.from_numpy(points).to(device)
    return points, edge_index


#############################################
##               Write Function            ##
#############################################

def generate_tsp_file(node_coords:np.ndarray, filename):
    """
    Generate a TSP problem data file based on the given point node_coords.

    Args:
        node_coords: A two-dimensional list containing the point node_coords, e.g. [[x1, y1], [x2, y2], ...]
        filename: The filename of the generated TSP problem data file

    """
    if node_coords.ndim == 3:
        shape = node_coords.shape
        if shape[0] == 1:
            node_coords = node_coords.squeeze(axis=0)
            _generate_tsp_file(node_coords,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_tsp_file(node_coords[i],_filename)
    else:
        assert node_coords.ndim == 2
        _generate_tsp_file(node_coords,filename)
             
             
def _generate_tsp_file(node_coords:np.ndarray, filename):
    num_points = node_coords.shape[0]
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_points):
            x, y = node_coords[i]
            f.write(f"{i+1} {x} {y}\n")
        f.write("EOF\n")


def generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    if tour.ndim == 2:
        shape = tour.shape
        if shape[0] == 1:
            tour = tour.squeeze(axis=0)
            _generate_opt_tour_file(tour,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_opt_tour_file(tour[i],_filename)
    else:
        assert tour.ndim == 1
        _generate_opt_tour_file(tour,filename)
   
   
def _generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    num_points = len(tour)
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write(f"TYPE: TOUR\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write(f"TOUR_SECTION\n")
        for i in range(num_points):
            f.write(f"{tour[i]}\n")
        f.write(f"-1\n")
        f.write(f"EOF\n")


def save_heatmap_txt(heatmap: np.ndarray, filename: str, save_path: str="heatmap"):
    if heatmap.ndim == 2:
        _save_heatmap(heatmap, filename+"_0", save_path)
    else:
        for i in range(heatmap.shape[0]):
            _save_heatmap(heatmap[i], filename+"_{}".format(i), save_path)


def _save_heatmap(heatmap: np.ndarray, filename: str, save_path: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num_nodes = heatmap.shape[1]
    filename = os.path.join(save_path, filename)
    np.savetxt(filename, heatmap, fmt = '%.6f', delimiter = ' ', header = f'{num_nodes}', comments = '')
    
    
#############################################
##         get data/tour from files        ##
#############################################

def get_data_from_tsp_file(filename):
    tsp_data = tsplib95.load(filename)
    if tsp_data.node_coords == {}:
        num_nodes = tsp_data.dimension
        edge_weights = tsp_data.edge_weights
        edge_weights = [elem for sublst in edge_weights for elem in sublst]
        new_edge_weights = np.zeros(shape=(num_nodes,num_nodes))
        if (num_nodes * (num_nodes-1) / 2) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [0,0,1,1,1],
             [0,0,0,1,1],
             [0,0,0,0,1],
             [0,0,0,0,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i >= j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif (num_nodes * (num_nodes+1) / 2) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [0,x,1,1,1],
             [0,0,x,1,1],
             [0,0,0,x,1],
             [0,0,0,0,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i > j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif ((num_nodes-1) * (num_nodes-1)) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [1,0,1,1,1],
             [1,0,0,1,1],
             [1,1,1,0,1],
             [1,1,1,1,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
        elif (num_nodes * num_nodes) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [1,x,1,1,1],
             [1,0,x,1,1],
             [1,1,1,x,1],
             [1,1,1,1,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1  
        else:
            raise ValueError("edge_weights cannot form a Symmetric matrix") 
        data = TSP_DATA(edge_weights=new_edge_weights)
    else:
        node_coords = np.array(list(tsp_data.node_coords.values()))
        data = TSP_DATA(node_coords)
    return data


def get_tour_from_tour_file(filename):
    tsp_tour = tsplib95.load(filename)
    tsp_tour = np.array(tsp_tour.tours).squeeze(axis=0)
    return tsp_tour


#############################################
##                parameter                ##
#############################################

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print('\n')
    print(t.draw())


#############################################
##    coords matrix to distance matrix     ##
#############################################

def coords_to_distance(points: torch.Tensor, edge_index: torch.Tensor=None,):
    device = points.device
    if edge_index is None:
        if points.ndim == 2:
            distance_matrix = torch.cdist(points, points)
        else:
            distance_matrix = torch.zeros(size=(points.shape[0], points.shape[1], points.shape[1]))
            for i, matrix in enumerate(points):
                distance_matrix[i] = torch.cdist(matrix, matrix)
    else:     
        if points.ndim == 2:
            x = edge_index[0]
            y = edge_index[1]
            points_x = points[x]
            points_y = points[y]
            distance_matrix = torch.norm(points_x - points_y, dim=1)
        else:
            matrix_list = list()
            for i in range(points.shape[0]):
                x = edge_index[i][0]
                y = edge_index[i][1]
                points_x = points[i][x]
                points_y = points[i][y]
                matrix_list.append(torch.norm(points_x - points_y, dim=1))
            distance_matrix = torch.stack(matrix_list)
    return distance_matrix.to(device)
