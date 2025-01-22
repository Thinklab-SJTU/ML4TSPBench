import numpy as np
from typing import Union
from ml4co_kit import tsp_mcts_decoder
from scipy.spatial.distance import cdist
from ml4tsp.nar.decoder.base import ML4TSPNARDecoder


MCTS_TIME_LIMIT_DEFAULT = {
      50: 0.005,
     100: 0.020,
     200: 0.100,
     500: 1.000,
    1000: 5.000
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


class ML4TSPNARMCTSDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2,
        time_limit: Union[float, str] = "auto",
        time_multiplier: int = 1, 
        max_depth: int = 10,
        max_iterations_2opt: int = 5000,
        mcts_smooth: bool = False,
        type_2opt: int = 1,
    ):
        super(ML4TSPNARMCTSDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr, 
        )

        self.time_limit = time_limit
        self.time_multiplier = time_multiplier
        self.max_depth = max_depth
        self.max_iterations_2opt = max_iterations_2opt
        self.mcts_smooth = mcts_smooth
        self.type_2opt = type_2opt
    
    def smooth_heatmap(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        graph = np.array(cdist(points, points))
        graph = graph / graph.max(axis=1, keepdims=True)
        new_heatmap = np.exp(np.tan(graph) * np.log(heatmap))
        np.fill_diagonal(new_heatmap, 1e-14)
        new_heatmap = new_heatmap / new_heatmap.sum(axis=1, keepdims=True)
        new_heatmap = new_heatmap * 2
        return new_heatmap
    
    def _decode(self, heatmap: np.ndarray, points: np.ndarray = None) -> np.ndarray:
        # time_limit
        change_time_limit = False
        if self.time_limit == "auto":
            change_time_limit = True
            x = get_nodes_num_for_mcts_time_limit(self.nodes_num)
            self.time_limit = MCTS_TIME_LIMIT_DEFAULT[x]
        self.time_limit *= self.time_multiplier
        
        # decoding
        tours = tsp_mcts_decoder(
            heatmap=heatmap, points=points, time_limit=self.time_limit, 
            max_depth=self.max_depth, type_2opt=self.type_2opt, 
            max_iterations_2opt=self.max_iterations_2opt
        )

        # change type
        if change_time_limit:
            self.time_limit = "auto"

        return tours
    