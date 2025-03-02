from typing import Any
import pytorch_lightning as pl


class TSPNAREncoder(pl.LightningModule):
    def __init__(self):
        super(TSPNAREncoder, self).__init__()
        self.gap_list = list()
        self.decoding_time = 0
        self.ls_time = 0
        
    def solve(self, data: Any, batch_size: int=16, device='cpu'):
        """solve function, return heatmap"""
        raise NotImplementedError("solve is required to implemented in subclass")
    
    def load_ckpt(self):
        """load state dict from checkpoint"""
        raise NotImplementedError("load_ckpt is required to implemented in subclass")