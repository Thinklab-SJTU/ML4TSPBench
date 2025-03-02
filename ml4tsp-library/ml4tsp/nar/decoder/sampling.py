import numpy as np
from .base import ML4TSPNARDecoder


class ML4TSPNARSamplingDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2,
        samples_num: int = 10
    ):
        super(ML4TSPNARSamplingDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr, 
        )
        self.samples_num = samples_num
    
    def _decode(self, heatmap: np.ndarray, points: np.ndarray = None) -> np.ndarray:
        tours = list()
        for idx in range(self.batch_size):
            _tours = list()
            for _ in range(self.samples_num):
                _heatmap = heatmap[idx]
                _tour = np.zeros(self.nodes_num + 1, dtype=np.int64)
                mask = np.ones(shape=(self.nodes_num , self.nodes_num))
                mask[:, 0] = 0
                np.fill_diagonal(mask, 0)
                selected_num = 0
                selected_node = 0
                while(selected_num < self.nodes_num - 1):
                    _heatmap = _heatmap * mask
                    probs = _heatmap[selected_node] / np.sum(_heatmap[selected_node])
                    selected_node = np.random.choice(np.arange(len(probs)), p=probs)
                    selected_num = selected_num + 1
                    _tour[selected_num] = selected_node
                    mask[:, selected_node] = 0
                _tours.append(_tour)
            tours.append(_tours)
        tours = np.array(tours)
        return tours
    