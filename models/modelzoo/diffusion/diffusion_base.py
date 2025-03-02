import torch
import torch.utils.data
from models.modelzoo.base.nar import TSPNAREncoder
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info
from models.utils.diffusion import get_schedule_fn
from models.nn import GNNEncoder, SAGEncoder
from utils.utils import TSPGraphDataset
from typing import Any
import torch.nn.functional as F
import numpy as np

from models.utils.diffusion import get_schedule_fn, CategoricalDiffusion, InferenceSchedule


class MetaDiffusion(TSPNAREncoder):
    def __init__(
        self,
        num_nodes: int=50,
        # network
        network_type: str="gnn",
        input_dim: int=2,
        embedding_dim: int=128,
        hidden_dim: int=256,
        output_channels: int=2,
        num_layers: int=12,
        aggregation: str="sum", 
        norm: str="layer",
        learn_norm: bool=True, 
        track_norm: bool=False, 
        gated: bool=True,
        sparse_factor: int=-1, 
        use_activation_checkpoint: bool=False, 
        node_feature_only: bool=False, 
        time_embed_flag: bool=True,
        # train/valid/test
        train_batch_size: int=64,
        valid_batch_size: int=1,
        test_batch_size: int=1,
        train_file: str=None,
        valid_file: str=None,
        test_file: str=None,
        regret_dir: str=None,
        valid_samples: int=1280,
        mode: str=None,
        num_workers: int=0,
        # training params
        lr_scheduler: str="cosine-decay",
        learning_rate: float=2e-4,
        weight_decay: float=1e-4,
        # diffusion params
        diffusion_schedule: str="linear",
        inference_schedule: str="cosine",
        diffusion_steps: int=1000,
        inference_diffusion_steps: int=50,
        **kwargs
    ):
        super(MetaDiffusion, self).__init__()
        
        self.num_nodes = num_nodes
        # net & model
        self.net=self.get_net(network_type)
        self.sparse_factor=sparse_factor
        self.sparse=self.sparse_factor > 0
        self.model=self.net(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_layers=num_layers,
            aggregation=aggregation,
            norm=norm,
            learn_norm=learn_norm,
            track_norm=track_norm,
            gated=gated,
            sparse=self.sparse,
            use_activation_checkpoint=use_activation_checkpoint,
            node_feature_only=node_feature_only,
            time_embed_flag=time_embed_flag
        )
        
        # train/valid/test
        self.train_batch_size=train_batch_size
        self.valid_batch_size=valid_batch_size
        self.test_batch_size=test_batch_size
        self.train_file=train_file
        self.valid_file=valid_file
        self.test_file=test_file
        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None
        self.regret_dir=regret_dir
        self.valid_samples=valid_samples
        self.mode=mode
        self.num_workers=num_workers
        
        # training params
        self.lr_scheduler=lr_scheduler
        self.num_training_steps_cached=None
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay

        # diffusion params
        self.diffusion_schedule = diffusion_schedule
        self.inference_schedule = inference_schedule
        self.diffusion_steps = diffusion_steps
        self.inference_diffusion_steps = inference_diffusion_steps
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule)

        # load data
        self.load_data()
        
    def get_net(self, network_type: str):
        """
        Get Network
        """
        self.network_type = network_type
        if network_type == 'gnn':
            return GNNEncoder
        elif network_type == 'sag':
            return SAGEncoder
        else:
            raise ValueError(f"{network_type} is an unknown network type.")

    def change_mode(self, mode:str):
        self.mode = mode
        try:
            self.load_data()
        except:
            rank_zero_info(f"{mode} file is not provided.")
            pass

    def load_data(self):
        if self.mode == "train":
            self.train_dataset = TSPGraphDataset(
                data_file=self.train_file,
                sparse_factor=self.sparse_factor,
                regret_path=self.regret_dir
            )
            self.valid_dataset = TSPGraphDataset(
                data_file=self.valid_file,
                sparse_factor=self.sparse_factor,
                regret_path=self.regret_dir
            )
        elif self.mode == "test":
            self.test_dataset = TSPGraphDataset(
                data_file=self.test_file,
                sparse_factor=self.sparse_factor,
                regret_path=self.regret_dir
            )
        else:
            pass
        
    def get_total_num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.
        """
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        
        dataset=self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        
        dataset_size=(
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )
        num_devices=max(1, self.trainer.num_devices)
        effective_batch_size=self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached=(dataset_size // effective_batch_size) * self.trainer.max_epochs
        
        return self.num_training_steps_cached
    
    def configure_optimizers(self):
        """
        """
        rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

        if self.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            optimizer=torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            scheduler=get_schedule_fn(
                self.lr_scheduler, 
                self.get_total_num_training_steps()
            )(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        
    def train_dataloader(self):
        train_dataloader=GraphDataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataset=torch.utils.data.Subset(
            dataset=self.valid_dataset, 
            indices=range(self.valid_samples)
        )
        val_dataloader=GraphDataLoader(
            val_dataset, 
            batch_size=self.valid_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        return self.test_dataset
    
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        """Shared step between train/val/test. To be implemented in subclass"""
        raise NotImplementedError("Shared step is required to implemented in subclass")

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")
    
    def forward(self, x, graph, edge_index, t=None):
        return self.model(x, graph=graph, edge_index=edge_index, timesteps=t)
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
        See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion
        
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        if self.sparse:
            xt = xt.reshape(-1)
        return xt
  
    def guided_categorical_posterior(self, target_t, t, x0_pred_prob, xt, grad=None):
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with torch.no_grad():
            diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)  # [2, 2], transition matrix
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2), \
            torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
        xt_grad_zero[..., 0] = (1 - xt) * grad
        xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
        xt_grad_one[..., 1] = xt * grad
        xt_grad_one[..., 0] = -xt_grad_one[..., 1]
        xt_grad = xt_grad_zero + xt_grad_one

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

        # q(xt−1|xt,x0=0)pθ(x0=0|xt)
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3  # [b, n, n, 2]

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        # q(xt−1|xt,x0=1)pθ(x0=1|xt)
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim=-1)
        p_phi = torch.exp(-xt_grad)
        if self.sparse:
            p_phi = p_phi.reshape(p_theta.shape)
        posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim=-1, keepdim=True)

        if target_t > 0:
            xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
        else:
            xt = posterior[..., 1].clamp(min=0)
        if self.sparse:
            xt = xt.reshape(-1)
        return xt