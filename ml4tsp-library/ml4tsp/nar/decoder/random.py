import numpy as np
from .base import ML4TSPNARDecoder


class ML4TSPNARRandomDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2,
        samples_num: int = 10,
    ):
        super(ML4TSPNARRandomDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr, 
        )
        self.samples_num = samples_num

    def _decode(self, heatmap: np.ndarray = None, points: np.ndarray = None) -> np.ndarray:
        tours = list()
        for _ in range(self.batch_size):
            _tours = list()
            # random
            for _ in range(self.samples_num):
                tour = np.arange(1, self.nodes_num)
                np.random.shuffle(tour)
                tour = np.insert(tour, [0, len(tour)], [0, 0])
                _tours.append(tour)
            random_tours = np.array(_tours)
            tours.append(random_tours)
        return np.array(tours)