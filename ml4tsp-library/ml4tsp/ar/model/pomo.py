import torch
import numpy as np
from typing import Union, Any
from tensordict import TensorDict
from ml4co_kit import download
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.policy.pomo import ML4TSPPOMOPolicy
from ml4tsp.ar.baseline.baseline import REINFORCEBaseline
from ml4tsp.ar.model.base import ML4TSPARBaseModel
from ml4tsp.ar.local_search import ML4TSPARLocalSearch
from ml4tsp.ar.utils.ops import gather_by_index, get_num_starts, unbatchify
from ml4tsp.ar.utils.transforms import StateAugmentation


class ML4TSPPOMO(ML4TSPARBaseModel):
    """
    Attention Model based on REINFORCE.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        param_args: Keyword arguments passed to the superclass
    """
    def __init__(
        self,
        env: ML4TSPAREnv,
        policy: ML4TSPPOMOPolicy,
        baseline: Union[REINFORCEBaseline, str] = "shared",
        baseline_kwargs: dict = {},
        local_search: Union[ML4TSPARLocalSearch, str] = None,
        local_search_kwargs: dict = {},
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None,
        # special
        num_augment: int = 8,
        use_dihedral_8: bool = True,
        num_starts: int = None,
    ):
        # super
        super(ML4TSPPOMO, self).__init__(
            env=env, 
            policy=policy, 
            baseline=baseline,
            baseline_kwargs=baseline_kwargs,
            local_search=local_search,
            local_search_kwargs=local_search_kwargs,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # env
        self.env: ML4TSPAREnv

        # policy
        self.model: ML4TSPPOMOPolicy
        
        # special
        self.num_starts = num_starts
        self.num_augment = num_augment
        self.augment = StateAugmentation(
            env_name="tsp", 
            num_augment=self.num_augment, 
            use_dihedral_8=use_dihedral_8
        )
        
        # load weights
        if pretrained and pretrained_path is not None:
            self.load_weights(pretrained_path)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.model(td, self.env, phase=phase, num_starts=n_start)
        out: dict
        
        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_start, n_aug))
            self.calculate_loss(td, batch, out, reward, log_likelihood)

        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=1)
                out.update({"max_reward": max_reward})

                # Reshape batch to [batch, n_start, n_aug]
                if out.get("actions", None) is not None:
                    actions = unbatchify(out["actions"], (n_start, n_aug))
                    out.update(
                        {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})
                if out.get("best_multistart_actions", None) is not None:
                    out.update(
                        {
                            "best_aug_actions": gather_by_index(
                                out["best_multistart_actions"], max_idxs
                            )
                        }
                    )
        metrics = self.get_metrics(out, phase)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return {"loss": out.get("loss", None), **metrics}

    def load_weights(
        self, pretrained_path: str, load_baseline: bool = False
    ):
        state_dict: dict = torch.load(pretrained_path, map_location="cpu")
        if load_baseline:
            self.load_state_dict(state_dict)
        else:
            model_state_dict = dict()
            for key, value in state_dict.items():
                if "baseline" not in key:
                    model_state_dict[key] = value
            self.load_state_dict(model_state_dict)
        self.to(self.env.device)
        
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
        
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td) if n_start is None else n_start

        # Evaluate policy
        actions = self.model(td, self.env, phase="test", num_starts=n_start)[1]
        
        return self.get_tours_from_actions(actions, batch_size)
    