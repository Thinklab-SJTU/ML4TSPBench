import os
import time
import torch
import random
import string
import numpy as np

from tqdm import tqdm
from typing import Any, Optional, Union
from tensordict import TensorDict

from lightning_utilities.core.rank_zero import rank_zero_info

from .models_base import TSPAREncoder
from .trainer import RL4COTrainer
from .reinforce_baseline import REINFORCEBaseline, get_reinforce_baseline
from models.rl4co.utils import get_lightning_device
from models.rl4co.envs import RL4COEnvBase
from models.rl4co.policy import AutoregressivePolicy
from models.rl4co.search import EASEmb, EASLay
from search import get_local_search_func
from utils.evaluator import TSPEvaluator


class REINFORCE(TSPAREncoder):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `TSPAREncoder` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        num_nodes:int,
        env: RL4COEnvBase,
        policy: AutoregressivePolicy,
        mode: str=None,
        train_batch_size: int=640,
        valid_batch_size: int=640,
        test_batch_size: int=None,
        train_data_size: int=1280000,
        valid_file: str=None,
        test_file: str=None,
        num_workers: int=0,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs={},
        eas_type: str=None,
        eas_batch_size: int=2,
        eas_max_iters: int=200,
        eas_filename: str=None,
        eas_save_file: bool=False,
        local_search_type: str=None,
        **kwargs
    ):

        super().__init__(
            num_nodes=num_nodes,
            env=env, 
            policy=policy,
            mode=mode,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=test_batch_size,
            train_data_size=train_data_size,
            valid_file=valid_file,
            test_file=test_file,
            num_workers=num_workers,
            eas_type=eas_type,
            eas_batch_size=eas_batch_size,
            eas_max_iters=eas_max_iters,
            eas_filename=eas_filename,
            eas_save_file=eas_save_file,
            local_search_type=local_search_type,
            **kwargs
        )
        
        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                rank_zero_info("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
     
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)
        
        metrics = self.get_metrics(out, phase)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return {"loss": out.get("loss", None), **metrics}
    
    def test_step(self, batch: Any, batch_idx: int):
        np_points = batch['locs'].cpu().detach().numpy()
        np_gt_tours = batch['gt_tour'].cpu().detach().numpy() - 1
        # Gain the datasets
        start_time = time.time()

        # EAS or Normal Solve
        if self.eas_type is not None:
            actions = self.eas()
        else:
            td = self.env.reset(batch)
            if self.test_batch_size == 1:
                td.batch_size = torch.Size([])
            _, actions, _ = self.policy(td, self.env, phase='test')

        # Gain the solution
        actions: torch.Tensor
        np_actions = actions.cpu().detach().numpy()
        np_actions: np.ndarray
        tours = list()
        if np_actions.ndim == 1:
            np_actions = np.expand_dims(np_actions, axis=0)
        for tour in np_actions:
            tour: np.ndarray
            zero_index = tour.tolist().index(0)
            new_tour = np.concatenate([tour[zero_index:], tour[:zero_index]])
            tours.append(new_tour)
        np_tours = np.array(tours)
        np_tours = np.append(np_tours, np_tours[:, :1], axis=1)
        
        if "multistart" in self.policy.test_decode_type:
            np_tours = np_tours.reshape(self.num_nodes, -1, np_tours.shape[-1]).transpose(1, 0, 2)
        end_time = time.time()
        decoding_time = end_time-start_time
        print(f"Decoding Time: {decoding_time}")

        # local_search
        if self.local_search_type is not None:
            start_time = time.time()
            self.local_search_func = get_local_search_func(task="tsp", name=self.local_search_type)
            ls_tours = list()
            for idx in tqdm(range(np_tours.shape[0]), desc='Local Search'):
                tour = self.local_search_func(
                    np_points=np_points[idx],
                    tours=np_tours[idx],
                    adj_mat=None,
                    device='cpu',
                    **self.ls_kwargs
                )
                ls_tours.append(tour)
            np_tours = np.array(ls_tours)
            end_time = time.time()
            ls_time = end_time-start_time
            print(f"Local Search Time: {ls_time}")
            total_time = ls_time + decoding_time
            print(f"Total Time: {total_time}")

        # caculate gap
        gap_total=0
        solved_cost_total=0
        gt_cost_total=0
        samples = np_tours.shape[0]
        for idx in range(samples):
            eva = TSPEvaluator(np_points[idx])
            gt_cost = eva.evaluate(np_gt_tours[idx])
            solved_tours = np_tours[idx]
            solved_tours: np.ndarray
            if solved_tours.ndim == 1:
                best_solved_cost = eva.evaluate(solved_tours)
            else:
                all_solved_costs = list()
                for tour in solved_tours:
                    all_solved_costs.append(eva.evaluate(tour))
                best_solved_cost = np.min(all_solved_costs)
            gap = (best_solved_cost - gt_cost) / gt_cost * 100
            self.gap_list.append(gap)
            gap_total += gap
            solved_cost_total += best_solved_cost
            gt_cost_total += gt_cost
        
        gap_avg = gap_total / samples
        solved_cost_avg = solved_cost_total / samples
        gt_cost_avg = gt_cost_total / samples

        metrics = {
            f"test/gt_cost": gt_cost_avg,
            f"test/gap": gap_avg,
            f"test/solved_cost": solved_cost_avg
        }
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        return metrics

    def eas(self):
        self.train()
        # params for eas
        device = get_lightning_device(self)
        eas_policy = self.policy.to(device)
        batch_size = self.eas_batch_size
        dataset = self.test_dataset
        max_iters=self.eas_max_iters
        
        # save path
        if not os.path.exists("eas_files"):
            os.mkdir("eas_files")
        if self.eas_filename is None:
            characters = string.ascii_letters + string.digits
            self.eas_filename = ''.join(random.choice(characters) for _ in range(6)) + '.pt'
        eas_save_path = os.path.join("eas_files", self.eas_filename)
        
        # eas model
        if self.eas_type == "eas_emb":
            eas_model = EASEmb(self.env, eas_policy, dataset, batch_size=batch_size, 
                               max_iters=max_iters, save_path=eas_save_path)
        elif self.eas_type == "eas_lay":
            eas_model = EASLay(self.env, eas_policy, dataset, batch_size=batch_size, 
                    max_iters=max_iters, save_path=eas_save_path)
        else:
            raise ValueError(f"{self.eas_type} is an illegal eas_type, the EAS model only supports 'eas_lay' and 'eas_emb'!")
        eas_model.setup()

        # train the eas model
        trainer = RL4COTrainer(
            max_epochs=1,
            gradient_clip_val=None,
        )
        trainer.fit(eas_model)
        
        # return the actions
        actions = torch.load(eas_save_path)["solutions"]
        actions = actions.reshape(self.test_data_size, -1)[:, 0:self.num_nodes]
        if not self.eas_save_file:
            os.remove(eas_save_path)
        return actions
            
    def solve(self, data:Union[np.ndarray, torch.Tensor], batch_size:int=None, device='cpu'):
        # set the policy to the mode 'eval'
        self.policy.eval()
        
        # deal with data
        if type(data) == np.ndarray:
            data = torch.Tensor(data)
        data: torch.Tensor # (B1, N, 2)
        if data.shape[0] == 1:
            data = data.unsqueeze(dim=0)

        # deal with different batch size
        # data: (S, B2, N, 2)
        if batch_size is None:
            samples = 1
            data = data.unsqueeze(dim=0)
        else:
            samples = data.shape[0] // batch_size
            assert samples * batch_size == data.shape[0], f"Batchsize cannot be divided by the quantity of data"
            data = data.reshape(-1, batch_size, data.shape[-2], data.shape[-1])
        
        # solve
        np_tours_list = list()
        for idx in tqdm(range(samples), desc='Decoding'):
            x_dict = {'locs': data[idx].float()}
            batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
            td = TensorDict(x_dict, batch_size=batch_size).to(device=device)
            td = self.env.reset(td)
            if td['locs'].shape[0] == 1:
                td.batch_size = torch.Size([])
            _, actions, _ = self.policy(td, self.env, phase='test')
            actions: torch.Tensor
            np_actions = actions.cpu().detach().numpy()
            np_actions: np.ndarray
            tours = list()
            if np_actions.ndim == 1:
                np_actions = np.expand_dims(np_actions, axis=0)
            for tour in np_actions:
                tour: np.ndarray
                zero_index = tour.tolist().index(0)
                new_tour = np.concatenate([tour[zero_index:], tour[:zero_index]])
                tours.append(new_tour)
            np_tours = np.array(tours)
            np_tours = np.append(np_tours, np_tours[:, :1], axis=1)
            if "multistart" in self.policy.test_decode_type:
                np_tours = np_tours.reshape(self.num_nodes, -1, np_tours.shape[-1]).transpose(1, 0, 2)
            np_tours_list.append(np_tours)
        if samples == 1:
            solved_tours = np_tours_list[0]
        else:
            solved_tours = np.concatenate(np_tours_list, axis=0) 
        return solved_tours

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
        if self.mode == 'train':
            dataset_size = self.val_data_size
        elif self.mode == 'test':
            dataset_size = self.test_data_size
        else:
            dataset_size = 1280
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.valid_batch_size,
            device=get_lightning_device(self),
            dataset_size=dataset_size,
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        rank_zero_info("the current train epoch is finished")
        rank_zero_info("Begin: Callback for end of training epoch: we evaluate the baseline...")
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.valid_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.val_data_size,
        )
        rank_zero_info("End: Callback for end of training epoch: we evaluate the baseline...")
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.valid_batch_size,
            device=get_lightning_device(self),
        )

    def set_decode_type_multistart(self, phase: str):
        """Set decode type to `multistart` for train, val and test in policy.
        For example, if the decode type is `greedy`, it will be set to `greedy_multistart`.

        Args:
            phase: Phase to set decode type for. Must be one of `train`, `val` or `test`.
        """
        attribute = f"{phase}_decode_type"
        attr_get = getattr(self.policy, attribute)
        # If does not exist, log error
        if attr_get is None:
            rank_zero_info(f"Decode type for {phase} is None. Cannot add `_multistart`.")
            return
        elif "multistart" in attr_get:
            return
        else:
            setattr(self.policy, attribute, f"{attr_get}_multistart")
            
    def __repr__(self):
        message = f"env={self.env}, "
        message += f"policy={self.policy}, "
        message += f"baseline={self.baseline}"
        return f"{self.__class__.__name__}({message})"