import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple
from ml4tsp.nar.env import ML4TSPNAREnv
from ml4tsp.nar.decoder import ML4TSPNARDecoder
from ml4tsp.nar.model.base import ML4TSPNARBaseModel
from ml4tsp.nar.local_search import ML4TSPNARLocalSearch
from ml4tsp.nar.encoder.gnn.gnn_encoder import GNNEncoder


class ML4TSPGNNREG(ML4TSPNARBaseModel):
    def __init__(
        self,
        env: ML4TSPNAREnv,
        encoder: GNNEncoder,
        decoder: Union[ML4TSPNARDecoder, str] = "greedy",
        local_search: Union[ML4TSPNARLocalSearch, str] = None,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        # super
        super(ML4TSPGNNREG, self).__init__(
            env=env,
            encoder=encoder,
            decoder=decoder,
            local_search=local_search,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            pretrained_path=pretrained_path
        )
    
    def inference_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        assert not self.env.sparse, "Sparse not supported for GNNREG"
        
        with torch.no_grad():
            x0_pred = self.forward(
                x=points, graph=distmat, edge_index=edge_index, timesteps=None
            )

        reg_mat = x0_pred.squeeze()
        mask = self.env.regret_mask
        scaler = self.env.reg_scaler
        edge_reg = torch.masked_select(reg_mat, mask.to(reg_mat.device)).view(-1, 1) # shape: [# edges, 1]
        reg_pred = scaler.inverse_transform(edge_reg.cpu().numpy())
        reg_mat[mask] = torch.from_numpy(reg_pred.reshape(-1)).to(reg_mat.device)
        reg_mat = torch.triu(reg_mat, diagonal=1) + torch.tril(reg_mat.t(), diagonal=-1)
        # heatmap = 1 - (reg_mat / reg_mat.max()).unsqueeze(dim=0).cpu().numpy()
        heatmap = -1 * reg_mat.unsqueeze(dim=0).cpu().numpy() # for decoding consistency, i.e., larger the better.
        
        if self.env.mode == "solve":
            return heatmap
    
        # return loss if mode = `val`
        loss_func = nn.MSELoss()
        loss = loss_func(torch.triu(x0_pred.squeeze(1), diagonal=1), ground_truth.triu(diagonal=1))       
        return loss, heatmap
    
    def train_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor,
    ) -> Tensor:
        # x0_pred
        x0_pred = self.forward(
            x=points, graph=distmat, edge_index=edge_index # , t=None
        )
        reg_mat = torch.triu(x0_pred.squeeze(1), diagonal=1)

        # loss
        loss_func = nn.MSELoss()
        loss = loss_func(reg_mat, ground_truth.triu(diagonal=1))
        
        return loss