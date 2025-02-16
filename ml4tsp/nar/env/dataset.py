import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ml4co_kit import TSPSolver, to_tensor


class ML4TSPGraphDataset(Dataset):
    def __init__(self, file_path: str, mode: str, reg_path: str=None, reg_scaler=None, reg_mask=None):
        tmp_solver = TSPSolver()
        tmp_solver.from_txt(file_path=file_path, ref=True)
        self.points = tmp_solver.points
        self.ref_tours = tmp_solver.ref_tours
        self.reg_path = reg_path
        self.reg_scaler = reg_scaler
        self.reg_mask = reg_mask
        self.mode = mode
        
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        if self.reg_path is not None:
            assert self.reg_scaler is not None and self.reg_mask is not None
            reg_idx = idx
            reg_file = os.path.join(self.reg_path, f'{self.mode}_{reg_idx}.npy')
            reg_matrix = torch.from_numpy(np.load(reg_file))
            edge_reg = torch.masked_select(reg_matrix, self.reg_mask).view(-1, 1) # shape: [# edges, 1]
            reg_transformed = self.reg_scaler.transform(edge_reg.numpy())
            reg_matrix[self.reg_mask] = torch.from_numpy(reg_transformed.reshape(-1))
            return (
                to_tensor(self.points[idx]).float(),
                to_tensor(self.ref_tours[idx]).long(),
                reg_matrix.float(),
            )
        else:
            return (
                to_tensor(self.points[idx]).float(),
                to_tensor(self.ref_tours[idx]).long(),
            )
        