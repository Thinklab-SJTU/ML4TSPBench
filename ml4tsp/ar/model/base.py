import torch
import numpy as np
import pytorch_lightning as pl
from tensordict import TensorDict
from typing import Any, Optional, Union
from pytorch_lightning.utilities import rank_zero_info
from ml4co_kit import BaseModel, to_numpy
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.policy.base import ML4TSPARPolicy
from ml4tsp.ar.baseline.baseline import REINFORCEBaseline, get_reinforce_baseline
from ml4tsp.ar.local_search import ML4TSPARLocalSearch, get_local_search_by_name


class ML4TSPARBaseModel(BaseModel):
    def __init__(
        self,
        env: ML4TSPAREnv,
        policy: ML4TSPARPolicy,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs: dict = {},
        local_search: Union[ML4TSPARLocalSearch, str] = None,
        local_search_kwargs: dict = {},
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4
    ):
        # super
        super(ML4TSPARBaseModel, self).__init__(
            env=env,
            model=policy,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # env
        self.env: ML4TSPAREnv

        # baseline
        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        self.baseline = baseline
        
        # local search
        if local_search is not None:
            if type(local_search) == str:
                local_search_kwargs.update({"nodes_num": self.env.nodes_num})
                local_search = get_local_search_by_name(local_search)(**local_search_kwargs)
        self.local_search = local_search
        
        # need to log
        self.train_metrics = ["loss", "reward"]
        self.val_metrics = ["reward"]
        self.test_metrics = ["reward"]

    def setup(self, stage="fit"):
        # load data
        self.env.load_data()
        # post setup
        self.post_setup_hook()

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )
        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out
    
    def post_setup_hook(self):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.model,
            self.env,
            batch_size=self.env.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.env.val_data_size,
        )    
      
    def get_metrics(self, metric_dict: dict, phase: str):
        """Log metrics to logger and progress bar"""
        need_to_log = getattr(self, f"{phase}_metrics")
        metrics = {f"{phase}/{k}": v.mean() for k, v in metric_dict.items() if k in need_to_log}
        return metrics
    
    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        rank_zero_info("the current train epoch is finished")
        rank_zero_info("Begin: Callback for end of training epoch: we evaluate the baseline...")
        self.baseline.epoch_callback(
            self.model,
            env=self.env,
            batch_size=self.env.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.env.val_data_size,
        )
        rank_zero_info("End: Callback for end of training epoch: we evaluate the baseline...")
        
        # reset the dataset
        ori_mode = self.env.mode
        self.env.mode = "train"
        self.env.load_data()
        self.env.train_dataset = self.wrap_dataset(self.env.train_dataset)
        self.env.mode = ori_mode

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.env.val_batch_size,
            device=get_lightning_device(self),
        )
    
    def get_tours_from_actions(self, actions: torch.Tensor, batch_size: int) -> np.ndarray:
        # actions
        if actions.ndim == 1:
            actions = actions.unsqueeze(dim=0)
        actions = to_numpy(actions)
        
        # get tours
        tours = list()
        for tour in actions:
            zero_index = tour.tolist().index(0)
            new_tour = np.concatenate([tour[zero_index:], tour[:zero_index]])
            new_tour = np.append(new_tour, 0)
            tours.append(new_tour)    
        
        # reshape
        tours = np.array(tours)
        n = self.env.nodes_num
        tours = tours.reshape(tours.shape[0] // batch_size, batch_size, n+1)
        tours = tours.transpose(1, 0, 2).reshape(-1, n+1)
        tours = tours.reshape(-1, n+1) # (B*P, N+1)
        return tours
    
    def solve(self, points: torch.Tensor) -> np.ndarray:
        """
        Args:
            points (torch.Tensor): (B, N, 2)
        """
        # process data torch.Tensor -> TensorDict
        batch_size = points.shape[0]
        td = TensorDict({'locs': points}, batch_size=batch_size)
        td = td.to(device=points.device)
        td = self.env.reset(td)   
        
        # solve
        actions = self.model(td, self.env, phase='test')[1] # (B, N)
        return self.get_tours_from_actions(actions, batch_size)
    
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # env mode
        self.env.mode = phase
        
        # mode & batch_size
        td = self.env.reset(batch)
                
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.model(td, self.env, phase=phase)
        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)
        
        metrics = self.get_metrics(out, phase)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return {"loss": out.get("loss", None), **metrics}
    
    
def get_lightning_device(lit_module: pl.LightningModule) -> torch.device:
    """Get the device of the Lightning module before setup is called
    See device setting issue in setup https://github.com/Lightning-AI/lightning/issues/2638
    """
    try:
        if lit_module.trainer.strategy.root_device != lit_module.device:
            return lit_module.trainer.strategy.root_device
        return lit_module.device
    except Exception:
        return lit_module.device