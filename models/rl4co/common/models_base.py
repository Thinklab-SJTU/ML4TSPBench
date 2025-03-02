from functools import partial
from typing import Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_info

from models.rl4co.utils import create_optimizer, create_scheduler
from models.rl4co.data import tensordict_collate_fn, load_txt_to_tensordict, TensorDictDataset
from models.rl4co.envs import RL4COEnvBase
from models.rl4co.policy import AutoregressivePolicy


class TSPAREncoder(pl.LightningModule):
    """Base class for Lightning modules for RL4CO. This defines the general training loop in terms of
    RL algorithms. Subclasses should implement mainly the `shared_step` to define the specific
    loss functions and optimization routines.
    """

    def __init__(
        self, 
        num_nodes:int,
        env: RL4COEnvBase,
        policy: AutoregressivePolicy,
        # train/valid/test
        mode: str=None,
        train_batch_size: int=640,
        valid_batch_size: int=640,
        test_batch_size: int=None,
        train_data_size: int=1280000,
        valid_file: str=None,
        test_file: str=None,
        num_workers: int=0,
        # eas
        eas_type: str=None,
        eas_batch_size: int=2,
        eas_max_iters: int=200,
        eas_filename: str=None,
        eas_save_file: bool=False,
        # local search
        local_search_type: str=None,
        # training params
        optimizer: Union[str, torch.optim.Optimizer, partial] = "Adam",
        optimizer_kwargs: dict = {"lr": 1e-4},
        lr_scheduler: Union[str, torch.optim.lr_scheduler.LRScheduler, partial] = None,
        lr_scheduler_kwargs: dict = {
            "milestones": [80, 95],
            "gamma": 0.1,
        },
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "val/reward",
        **kwargs
    ):
        super(TSPAREncoder, self).__init__()

        self.num_nodes = num_nodes
        self.env = env
        self.policy = policy
        self.mode = mode
        self.train_metrics = ["loss", "reward"]
        self.val_metrics = ["reward"]
        self.test_metrics = ["reward"]
        
        # train/valid/test
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        
        self.train_data_size = train_data_size
        self.val_data_size = None
        self.test_data_size = None
        self.valid_file = valid_file
        self.test_file = test_file
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        self.num_workers = num_workers

        # eas
        self.eas_type = eas_type
        self.eas_batch_size = eas_batch_size
        self.eas_max_iters = eas_max_iters
        self.eas_filename = eas_filename
        self.eas_save_file = eas_save_file
        
        # local search
        self.local_search_type = local_search_type
        self.ls_kwargs = kwargs

        # training params
        self._optimizer_name_or_cls: Union[str, torch.optim.Optimizer] = optimizer
        self.optimizer_kwargs: dict = optimizer_kwargs
        self._lr_scheduler_name_or_cls: Union[
            str, torch.optim.lr_scheduler.LRScheduler
        ] = lr_scheduler
        self.lr_scheduler_kwargs: dict = lr_scheduler_kwargs
        self.lr_scheduler_interval: str = lr_scheduler_interval
        self.lr_scheduler_monitor: str = lr_scheduler_monitor
        
        # record gap
        self.gap_list = list()
        
    def setup(self, stage="fit"):
        """Base LightningModule setup method. This will setup the datasets and dataloaders

        Note:
            We also send to the loggers all hyperparams that are not `nn.Module` (i.e. the policy).
            Apparently PyTorch Lightning does not do this by default.
        """        
        if self.mode == 'train':
            self.train_dataset = self.wrap_dataset(
                self.env.dataset(self.train_data_size, phase="train")
            )
            val_data, val_data_size = load_txt_to_tensordict(self.valid_file, return_data_size=True)
            self.val_dataset = TensorDictDataset(val_data)
            self.val_data_size = val_data_size
        elif self.mode == 'test':
            test_data, test_data_size = load_txt_to_tensordict(self.test_file, return_data_size=True)
            self.test_dataset = TensorDictDataset(test_data)
            self.test_data_size = test_data_size
            if self.test_batch_size is None:
                self.test_batch_size = test_data_size
        self.post_setup_hook()
        
    def post_setup_hook(self):
        """Hook to be called after setup. Can be used to set up subclasses without overriding `setup`"""
        raise NotImplementedError("post_setup_hook is required to implemented in subclass")

    def configure_optimizers(self, parameters=None):
        """
        Args:
            parameters: parameters to be optimized. If None, will use `self.policy.parameters()
        """

        if parameters is None:
            parameters = self.policy.parameters()

        rank_zero_info(f"Instantiating optimizer <{self._optimizer_name_or_cls}>")
        if isinstance(self._optimizer_name_or_cls, str):
            optimizer = create_optimizer(
                parameters, self._optimizer_name_or_cls, **self.optimizer_kwargs
            )
        elif isinstance(self._optimizer_name_or_cls, partial):
            optimizer = self._optimizer_name_or_cls(parameters, **self.optimizer_kwargs)
        else:  # User-defined optimizer
            opt_cls = self._optimizer_name_or_cls
            optimizer = opt_cls(parameters, **self.optimizer_kwargs)
            assert isinstance(optimizer, torch.optim.Optimizer)

        # instantiate lr scheduler
        if self._lr_scheduler_name_or_cls is None:
            return optimizer
        else:
            rank_zero_info(f"Instantiating LR scheduler <{self._lr_scheduler_name_or_cls}>")
            if isinstance(self._lr_scheduler_name_or_cls, str):
                scheduler = create_scheduler(
                    optimizer, self._lr_scheduler_name_or_cls, **self.lr_scheduler_kwargs
                )
            elif isinstance(self._lr_scheduler_name_or_cls, partial):
                scheduler = self._lr_scheduler_name_or_cls(
                    optimizer, **self.lr_scheduler_kwargs
                )
            else:  # User-defined scheduler
                scheduler_cls = self._lr_scheduler_name_or_cls
                scheduler = scheduler_cls(optimizer, **self.lr_scheduler_kwargs)
                assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
            return [optimizer], {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
                "monitor": self.lr_scheduler_monitor,
            }

    def get_metrics(self, metric_dict: dict, phase: str):
        """Log metrics to logger and progress bar"""
        need_to_log = getattr(self, f"{phase}_metrics")
        metrics = {f"{phase}/{k}": v.mean() for k, v in metric_dict.items() if k in need_to_log}
        return metrics

    def forward(self, td, **kwargs):
        """Forward pass for the model. Simple wrapper around `policy`. Uses `env` from the module if not provided."""
        if kwargs.get("env", None) is None:
            env = self.env
        else:
            env = kwargs["env"]
        return self.policy(td, env, **kwargs)

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

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset, self.train_batch_size, True
        )

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.valid_batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size)

    def on_train_epoch_end(self):
        """Called at the end of the training epoch. This can be used for instance to update the train dataset
        with new data (which is the case in RL).
        """
        train_dataset = self.env.dataset(self.train_data_size, "train")
        self.train_dataset = self.wrap_dataset(train_dataset)

    def wrap_dataset(self, dataset):
        """Wrap dataset with policy-specific wrapper. This is useful i.e. in REINFORCE where we need to
        collect the greedy rollout baseline outputs.
        """
        return dataset

    def _dataloader(self, dataset, batch_size, shuffle=False):
        """The dataloader used by the trainer. This is a wrapper around the dataset with a custom collate_fn
        to efficiently handle TensorDicts.
        """
        data = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=tensordict_collate_fn,
        )
        return data

    def solve(self, data:Union[np.ndarray, torch.Tensor]):
        """solve function, return heatmap"""
        raise NotImplementedError("solve is required to implemented in subclass")
    
    def load_ckpt(self, ckpt_path:str=None):
        """load state dict from checkpoint"""
        raise NotImplementedError("load_ckpt is required to implemented in subclass")
