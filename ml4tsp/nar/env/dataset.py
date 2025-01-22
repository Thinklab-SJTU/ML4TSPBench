from torch.utils.data import Dataset
from ml4co_kit import TSPSolver, to_tensor


class ML4TSPGraphDataset(Dataset):
    def __init__(self, file_path: str):
        tmp_solver = TSPSolver()
        tmp_solver.from_txt(file_path=file_path, ref=True)
        self.points = tmp_solver.points
        self.ref_tours = tmp_solver.ref_tours
        
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        return (
            to_tensor(self.points[idx]).float(),
            to_tensor(self.ref_tours[idx]).long(),
        )