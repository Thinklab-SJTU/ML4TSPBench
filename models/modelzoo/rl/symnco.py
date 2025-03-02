import torch
from typing import Any
from models.rl4co.common import REINFORCE, AutoregressiveDecoder, GraphAttentionEncoder
from models.rl4co.envs import TSPEnv
from models.rl4co.utils import gather_by_index, get_num_starts, unbatchify
from models.rl4co.policy import invariance_loss,problem_symmetricity_loss, solution_symmetricity_loss, SymNCOPolicy
from models.rl4co.data import StateAugmentation
import pygmtools as pygm


tsp_symnco_path = {
    50: 'https://huggingface.co/ML4TSPBench/SYMNCO/resolve/main/tsp50_symnco.pt?download=true',
    100: 'https://huggingface.co/ML4TSPBench/SYMNCO/resolve/main/tsp100_symnco.pt?download=true',
    500: 'https://huggingface.co/ML4TSPBench/SYMNCO/resolve/main/tsp500_symnco.pt?download=true'
}


class TSPSymNCO(REINFORCE):
    """SymNCO Model based on REINFORCE with shared baselines.
    Based on Kim et al. (2022) https://arxiv.org/abs/2205.13209.

    Args:
        env: TorchRL environment to use for the algorithm
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        num_augment: Number of augmentations
        alpha: weight for invariance loss
        beta: weight for solution symmetricity loss
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        **kwargs: Keyword arguments passed to the superclass
    """
    def __init__(
        self,
        num_nodes: int=50,
        # symnco
        num_augment: int = 4,
        alpha: float = 0.2,
        beta: float = 1,
        num_starts: int = 0,
        # network
        hidden_dim: int=128,
        num_layers: int=3,
        num_heads: int=8,
        # train/valid/test
        mode: str=None,
        train_batch_size: int=640,
        valid_batch_size: int=640,
        train_data_size: int=1280000,
        valid_file: str=None,
        test_file: str=None,
        train_decode_type: str="sampling",
        val_decode_type: str=None,
        test_decode_type: str=None,
        decoding_type: str="greedy",
        num_workers: int=0,
        # local_search
        local_search_type: str=None,
        **kwargs
    ):
        
        self.num_nodes = num_nodes
        env = TSPEnv(num_loc=num_nodes)
        
        encoder = GraphAttentionEncoder(
            env_name=env.name,
            num_heads=num_heads,
            embedding_dim=hidden_dim,
            num_layers=num_layers,
        )

        decoder = AutoregressiveDecoder(
            env_name=env.name,
            embedding_dim=hidden_dim,
            num_heads=num_heads,
        )
        
        policy = SymNCOPolicy(
            env_name=env.name, 
            embedding_dim=hidden_dim,
            encoder=encoder, 
            decoder=decoder, 
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            decoding_type=decoding_type
        )  
        
        super(TSPSymNCO, self).__init__(
            num_nodes=num_nodes,
            env=env, 
            policy=policy, 
            mode=mode,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            train_data_size=train_data_size,
            valid_file=valid_file,
            test_file=test_file,
            num_workers=num_workers,
            baseline="no",
            local_search_type=local_search_type,
            **kwargs
        )

        self.num_starts = num_starts
        self.num_augment = num_augment
        self.augment = StateAugmentation(
            env_name=env.name, 
            num_augment=self.num_augment
        )
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss

        # Add `_multistart` to decode type for train, val and test in policy if num_starts > 1
        if self.num_starts > 1:
            for phase in ["train", "val", "test"]:
                self.set_decode_type_multistart(phase)

        self.setup()

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td) if n_start is None else n_start

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(td, self.env, phase=phase, num_starts=n_start)

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

    def load_ckpt(self, ckpt_path:str=None, load_baseline: bool=True):
        # load state_dict from pretrained checkpoint
        if ckpt_path is None:
            if self.num_nodes in [50, 100]:
                url = tsp_symnco_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_symnco.pt"
                pygm.utils.download(filename=filename, url=url, to_cache=None)
                state_dict = torch.load(filename)
            else:
                raise ValueError(f"There is currently no pretrained checkpoint with {self.num_nodes} nodes.")
        else:
            checkpoint = torch.load(ckpt_path)
            if ckpt_path.endswith('.ckpt'):
                state_dict = checkpoint['state_dict']
            elif ckpt_path.endswith('.pt'):
                state_dict = checkpoint
        # check if load the baseline
        state_dict: dict
        if load_baseline:
            self.load_state_dict(state_dict)
        else:
            policy_state_dict = dict()
            for key, value in state_dict.items():
                if not "baseline" in key:
                    policy_state_dict[key[7:]] = value
            self.policy.load_state_dict(policy_state_dict)