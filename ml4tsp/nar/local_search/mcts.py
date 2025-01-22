import ctypes
import numpy as np
import scipy.special as ssp
from typing import Union
from scipy.spatial.distance import cdist
from ml4co_kit import tsp_mcts_local_search
from ml4tsp.nar.local_search.base import ML4TSPNARLocalSearch


MCTS_TIME_LIMIT_DEFAULT = {
    "continue": {"continue_flag": True, 50: 0.005, 100: 0.020, 200: 0.100, 500: 1.000, 1000: 5.000},
    "fast": {"continue_flag": False, 50: 0.005, 100: 0.020, 200: 0.100, 500: 1.000, 1000: 5.000},
    "break": {"continue_flag": False, 50: 0.025, 100: 0.100, 200: 0.500, 500: 2.000, 1000: 10.000},
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


class ML4TSPNARMCTS(ML4TSPNARLocalSearch):
    def __init__(
        self,
        time_limit: Union[float, str] = "auto",
        time_multiplier: int = 1,
        max_depth: int = 10,
        max_iterations_2opt: Union[float, str] = "auto",
        mcts_smooth: bool = False,
        smooth_type: str = "v1",
        smooth_tau: float = 0.01,
        type_2opt: int = 2,
        type_mcts: Union[str, int] = "auto",
    ):
        super().__init__(time_limit=time_limit)
        self.max_depth = max_depth
        self.max_iterations_2opt = max_iterations_2opt
        self.mcts_smooth = mcts_smooth
        self.smooth_type = smooth_type
        self.smooth_func_dict = {
            "v1": self.smooth_heatmap,
            "v2": self.smooth_heatmap_v2,
            "v3": self.smooth_heatmap_v3
        }
        self.smooth_func = self.smooth_func_dict[self.smooth_type]
        self.smooth_tau = smooth_tau
        self.type_2opt = type_2opt
        self.type_mcts = type_mcts
        self.time_multiplier = time_multiplier
        
    def smooth_heatmap(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        graph = np.array(cdist(points, points))
        graph = graph / graph.max(axis=1, keepdims=True)
        new_heatmap = np.exp(np.tan(graph) * np.log(heatmap))
        np.fill_diagonal(new_heatmap, 1e-14)
        new_heatmap = new_heatmap / new_heatmap.sum(axis=1, keepdims=True)
        new_heatmap = new_heatmap * 2
        return new_heatmap
    
    def smooth_heatmap_v2(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        nodes_num = points.shape[-2]
        sorted_vector = np.sort(heatmap, axis=-1)[:, - nodes_num // 10].reshape(-1, 1)
        heatmap[(heatmap - sorted_vector) < 0] -= 1e9
        start = 1.0
        minimum = 0.0
        while minimum < 1e-4: # adjust temperature
            new_heatmap = ssp.softmax(heatmap * start, axis=-1)
            minimum = new_heatmap[new_heatmap > 0].min()
            start *= 0.5
        new_heatmap = new_heatmap / new_heatmap.sum(axis=1, keepdims=True)
        return new_heatmap
    
    def smooth_heatmap_v3(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        graph = 1.0 - np.array(cdist(points, points))
        new_heatmap = heatmap + self.smooth_tau * graph
        new_heatmap = new_heatmap / new_heatmap.sum(axis=1, keepdims=True)
        return new_heatmap
    
    def _local_search(
        self, tour: np.ndarray, points: np.ndarray, heatmap: np.ndarray = None
    ) -> np.ndarray:
        # smooth
        if self.mcts_smooth:
            heatmap = self.smooth_func(heatmap, points)

        # number of nodes
        self.nodes_num = heatmap.shape[-1]
        
        # continue_flag
        change_type_mcts = False
        if self.type_mcts == "auto":
            change_type_mcts= True
            self.type_mcts = "break" if self.nodes_num < 500 else "continue"
        continue_flag = MCTS_TIME_LIMIT_DEFAULT[self.type_mcts]["continue_flag"]
        continue_flag = 1 if continue_flag else 2

        # time_limit
        change_time_limit = False
        if self.time_limit == "auto":
            change_time_limit = True
            x = get_nodes_num_for_mcts_time_limit(self.nodes_num)
            self.time_limit = MCTS_TIME_LIMIT_DEFAULT[self.type_mcts][x]
        self.time_limit *= self.time_multiplier
        
        # max_iterations_2opt
        change_max_iterations_2opt = False
        if self.max_iterations_2opt == "auto":
            change_max_iterations_2opt = True
            if self.nodes_num < 200:
                self.max_iterations_2opt = 5000
            elif self.nodes_num < 500:
                self.max_iterations_2opt = 50
            else:
                self.max_iterations_2opt = 0
        
        # real local search
        mcts_tour = tsp_mcts_local_search(
            init_tours=tour, heatmap=heatmap, points=points,
            time_limit=self.time_limit, max_depth=self.max_depth,
            type_2opt=self.type_2opt, max_iterations_2opt=self.max_iterations_2opt
        )
        
        # change type
        if change_type_mcts:
            self.type_mcts = "auto"
        if change_time_limit:
            self.time_limit = "auto"
        if change_max_iterations_2opt:
            self.max_iterations_2opt = "auto"
        
        return mcts_tour