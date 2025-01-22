import numpy as np
from ml4co_kit import tsp_insertion_decoder
from ml4tsp.nar.decoder.base import ML4TSPNARDecoder


class ML4TSPNARInsertionDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2,
        samples_num: int = 10,
    ):
        super(ML4TSPNARInsertionDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr, 
        )
        self.samples_num = samples_num
        
    def _decode(self, heatmap: np.ndarray = None, points: np.ndarray = None) -> np.ndarray:
        tours = list()
        for _ in range(self.samples_num):
            tours.append(tsp_insertion_decoder(points=points))
        return np.array(tours)