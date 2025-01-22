import torch
from torch import nn, Tensor
from typing import Any, Union
from ml4co_kit import BaseModel, points_to_distmat
from ml4tsp.nar.env.env import ML4TSPNAREnv
from ml4tsp.nar.decoder import ML4TSPNARDecoder, get_nar_decoder_by_name
from ml4tsp.nar.local_search import ML4TSPNARLocalSearch, get_local_search_by_name


class ML4TSPNARBaseModel(BaseModel):
    def __init__(
        self,
        env: ML4TSPNAREnv,
        encoder: nn.Module,
        decoder: Union[ML4TSPNARDecoder, str] = "greedy",
        local_search: Union[ML4TSPNARLocalSearch, str] = None,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        # super
        super(ML4TSPNARBaseModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # env
        self.env: ML4TSPNAREnv
        
        # decoder
        if isinstance(decoder, str):
            decoder = get_nar_decoder_by_name(decoder)
        self.decoder = decoder
        self.decoder.device = self.env.device
        self.decoder.sparse = self.env.sparse
        
        # local search
        if local_search is not None:
            if isinstance(local_search, str):
                local_search = get_local_search_by_name(local_search)
                local_search.device = self.env.device
        self.local_search = local_search
        
        # load weights
        if pretrained and pretrained_path is not None:
            self.load_weights(pretrained_path)
        
    def set_mode(self, mode: str):
        if mode == "train":
            self.train()
        else:
            self.eval()
        self.model = self.model.to(self.env.device)
        self.env.mode = mode
    
    def set_nodes_num(self, nodes_num: int):
        self.env.nodes_num = nodes_num

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # env mode
        self.env.mode = phase
        
        # read data from batch
        points, ref_tours = batch
        points: Tensor # (B, N, 2)
        ref_tours: Tensor # (B, N+1)

        # process data
        ground_truth, edge_index = self.env.process_data(points, ref_tours)
        distmat = points_to_distmat(points, edge_index)
        
        # deal with different phase (mode)
        if phase == "train":
            loss = self.train_process(
                points=points, edge_index=edge_index, 
                distmat=distmat, ground_truth=ground_truth
            )
        elif phase == "val":
            loss, heatmap = self.inference_process(
                points=points, edge_index=edge_index, 
                distmat=distmat, ground_truth=ground_truth
            )
            costs_avg = self.decoder.decode(
                heatmap=heatmap, points=points, return_costs=True
            )
        else:
            raise NotImplementedError()
        
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # return
        return loss if phase == "train" else metrics   

    def train_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> Tensor:
        raise NotImplementedError(
            "``train_process`` is required to implemented in subclasses."
        )

    def inference_process(
        self, points: Tensor, edge_index: Tensor, distmat: Tensor, ground_truth: Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "``inference`` is required to implemented in subclasses."
        )

    def load_weights(self, pretrained_path: str):
        self.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        self.to(self.env.device)