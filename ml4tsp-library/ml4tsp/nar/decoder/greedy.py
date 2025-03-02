import numpy as np
from ml4co_kit import tsp_greedy_decoder
from .base import ML4TSPNARDecoder


class ML4TSPNARGreeyDecoder(ML4TSPNARDecoder):
    def __init__(
        self,
        heatmap_delta: float = 1e-14,
        active_search: bool = False,
        as_steps: int = 100,
        as_samples: int = 1000,
        as_inner_lr: float = 5e-2
    ):
        super(ML4TSPNARGreeyDecoder, self).__init__(
            heatmap_delta=heatmap_delta, active_search=active_search, 
            as_steps=as_steps, as_samples=as_samples, as_inner_lr=as_inner_lr, 
        )
    
    def _decode(self, heatmap: np.ndarray, points: np.ndarray) -> np.ndarray:
        tours = tsp_greedy_decoder(heatmap)
        if tours.ndim == 2:
            tours = np.expand_dims(tours, 0)
        return tours