import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Union, Tuple
from sklearn.utils.class_weight import compute_class_weight
from ..env import ML4TSPNAREnv
from ..decoder import ML4TSPNARDecoder
from ..model.base import ML4TSPNARBaseModel
from ..local_search import ML4TSPNARLocalSearch
from ..encoder.gnn.gnn_encoder import GNNEncoder


class ML4TSPGNNWISE(ML4TSPNARBaseModel):
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
        pretrained_path: str = None,
        wise_epsilon: float = 1e-14
    ):
        # super
        super(ML4TSPGNNWISE, self).__init__(
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
        self.wise_epsilon = wise_epsilon
    
    def inference_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.env.sparse:
            batch_size = points.shape[0]
            points = points.reshape(-1, 2)
            distmat = distmat.reshape(-1)
            edge_index = edge_index.transpose(1, 0).reshape(2, -1)
        
        with torch.no_grad():
            x0_pred = self.forward(
                x=points, graph=distmat, edge_index=edge_index, timesteps=None
            )
        
        if not self.env.sparse:
            x0 = F.softmax(x0_pred, dim=-1)
            x0_0 = (x0[:, 0, :, :] * (self.env.nodes_num - 2)).unsqueeze(dim=1)
            x0_1 = (x0[:, 1, :, :] * 2).unsqueeze(dim=1)
            h0 = torch.cat([x0_0, x0_1], dim=1)
            x0_pred = torch.clamp(h0, self.wise_epsilon, 1)
            heatmap = x0_pred[: , 1, :, :]
        else:
            x0_0_reshape = x0_pred[:, 0].reshape(-1, self.env.sparse_factor)
            x0_0_softmax = F.softmax(x0_0_reshape, dim=-1) * (self.env.sparse_factor - 2)
            x0_0 = x0_0_softmax.reshape(-1, 1)
            x0_1_reshape = x0_pred[:, 1].reshape(-1, self.env.sparse_factor)
            x0_1_softmax = F.softmax(x0_1_reshape, dim=-1) * 2
            x0_1 = x0_1_softmax.reshape(-1, 1)
            h0 = torch.cat([x0_0, x0_1], dim=1)
            x0_pred = torch.clamp(h0, self.wise_epsilon, 1)
            heatmap = x0_pred[: , 1]
            
        if self.env.mode == "solve":
            return heatmap
    
        # return loss if mode = `val`
        edge_labels = ground_truth.flatten().cpu().numpy()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edge_cw = torch.Tensor(edge_cw).to(x0_pred.device)
        log_x0_pred = torch.log(x0_pred)
        loss = nn.NLLLoss(edge_cw)(log_x0_pred, ground_truth.long())       
        return loss, heatmap
    
    def train_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Tensor:
        # x0_pred
        x0_pred = self.forward(
            x=points, graph=distmat, edge_index=edge_index, t=None
        )
        
        # wise
        if not self.env.sparse:
            x0 = F.softmax(x0_pred, dim=-1)
            x0_0 = (x0[:, 0, :, :] * (self.env.nodes_num - 2)).unsqueeze(dim=1)
            x0_1 = (x0[:, 1, :, :] * 2).unsqueeze(dim=1)
            h0 = torch.cat([x0_0, x0_1], dim=1)
            x0_pred = torch.clamp(h0, self.wise_epsilon, 1)
        else:
            x0_0_reshape = x0_pred[:, 0].reshape(-1, self.env.sparse_factor)
            x0_0_softmax = F.softmax(x0_0_reshape, dim=-1) * (self.env.sparse_factor - 2)
            x0_0 = x0_0_softmax.reshape(-1, 1)
            x0_1_reshape = x0_pred[:, 1].reshape(-1, self.env.sparse_factor)
            x0_1_softmax = F.softmax(x0_1_reshape, dim=-1) * 2
            x0_1 = x0_1_softmax.reshape(-1, 1)
            h0 = torch.cat([x0_0, x0_1], dim=1)
            x0_pred = torch.clamp(h0, self.wise_epsilon, 1)
        
        # loss
        edge_labels = ground_truth.flatten().cpu().numpy()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edge_cw = torch.Tensor(edge_cw).to(x0_pred.device)
        log_x0_pred = torch.log(x0_pred)
        loss = nn.NLLLoss(edge_cw)(log_x0_pred, ground_truth.long())
         
        return loss