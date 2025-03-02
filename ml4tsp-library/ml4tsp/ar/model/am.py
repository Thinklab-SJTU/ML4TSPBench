import torch
from typing import Union
from ml4co_kit import download
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.policy.am import ML4TSPAMPolicy
from ml4tsp.ar.baseline.baseline import REINFORCEBaseline
from ml4tsp.ar.model.base import ML4TSPARBaseModel
from ml4tsp.ar.local_search import ML4TSPARLocalSearch


class ML4TSPAM(ML4TSPARBaseModel):
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
        policy: ML4TSPAMPolicy,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs: dict = {},
        local_search: Union[ML4TSPARLocalSearch, str] = None,
        local_search_kwargs: dict = {},
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None
    ):
        # super
        super(ML4TSPAM, self).__init__(
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
        self.model: ML4TSPAMPolicy
        
        # load weights
        if pretrained and pretrained_path is not None:
            self.load_weights(pretrained_path)
            
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
                    model_state_dict[key.replace("policy.", "model.")] = value
            self.load_state_dict(model_state_dict, strict=True)
        self.to(self.env.device)
        