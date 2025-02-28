import torch
from typing import Any, Union
from ml4co_kit import download
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.baseline.baseline import REINFORCEBaseline
from ml4tsp.ar.model.base import ML4TSPARBaseModel
from ml4tsp.ar.utils.ops import gather_by_index, get_num_starts, unbatchify
from ml4tsp.ar.utils.transforms import StateAugmentation
from ml4tsp.ar.policy.symnco import invariance_loss, \
    problem_symmetricity_loss, solution_symmetricity_loss, ML4TSPSymNCOPolicy
from ml4tsp.ar.local_search import ML4TSPARLocalSearch 


class ML4TSPSymNCO(ML4TSPARBaseModel):
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
        # custom
        env: ML4TSPAREnv = None,
        policy: ML4TSPSymNCOPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "no",
        baseline_kwargs: dict = {},
        local_search: Union[ML4TSPARLocalSearch, str] = None,
        local_search_kwargs: dict = {},
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        pretrained_path: str = None,
        # special
        num_augment: int = 4,
        alpha: float = 0.2,
        beta: float = 1,
        num_starts: int = 0,
    ):
        # super
        super(ML4TSPSymNCO, self).__init__(
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
        self.model: ML4TSPSymNCOPolicy

        # special
        self.num_augment = num_augment
        self.alpha = alpha
        self.beta = beta
        self.num_starts = num_starts
        self.augment = StateAugmentation(
            env_name="tsp", 
            num_augment=self.num_augment
        )
        
        # load weights
        if pretrained and pretrained_path is not None:
            self.load_weights(pretrained_path)
            
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td) if n_start is None else n_start

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.model(td, self.env, phase=phase, num_starts=n_start)
        out: dict
        
        # Unbatchify reward to [batch_size, n_start, n_aug].
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # Main training loss
        if phase == "train":
            # [batch_size, n_start, n_aug]
            ll = unbatchify(out["log_likelihood"], (n_start, n_aug))

            # Calculate losses: problem symmetricity, solution symmetricity, invariance
            loss_ps = problem_symmetricity_loss(reward, ll) if n_start > 1 else 0
            loss_ss = solution_symmetricity_loss(reward, ll) if n_aug > 1 else 0
            loss_inv = invariance_loss(out["proj_embeddings"], n_aug) if n_aug > 1 else 0
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv
            out.update(
                {
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                }
            )

        # Log only during validation and test
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