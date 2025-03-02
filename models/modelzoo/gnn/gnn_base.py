import torch
import torch.utils.data
from models.modelzoo.base.nar import TSPNAREncoder
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info
from models.utils.diffusion import get_schedule_fn
from models.nn import GNNEncoder, SAGEncoder
from utils.utils import TSPGraphDataset
from typing import Any


class MetaGNN(TSPNAREncoder):
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
        time_embed_flag: bool=False,
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
        **kwargs
    ):
        super(MetaGNN, self).__init__()
        
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
                regret_path=self.regret_dir,
                mode='train'
            )
            self.valid_dataset = TSPGraphDataset(
                data_file=self.valid_file,
                sparse_factor=self.sparse_factor,
                regret_path=self.regret_dir,
                mode='val'
            )
        elif self.mode == "test":
            self.test_dataset = TSPGraphDataset(
                data_file=self.test_file,
                sparse_factor=self.sparse_factor,
                regret_path=self.regret_dir,
                mode='test'
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
        # force the test batch size to be 1
        self.test_batch_size = 1
        test_dataloader=GraphDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size, 
        )
        return test_dataloader
    
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