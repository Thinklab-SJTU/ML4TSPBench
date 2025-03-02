import torch
from typing import Union
from models.rl4co.common import REINFORCE, REINFORCEBaseline, AutoregressiveDecoder, GraphAttentionEncoder
from models.rl4co.envs import TSPEnv
from models.rl4co.policy import AttentionModelPolicy
import pygmtools as pygm


tsp_am_path = {
    50:  'https://huggingface.co/Bench4CO/AM/resolve/main/TSP/tsp50_am.ckpt?download=true',
    100: 'https://huggingface.co/Bench4CO/AM/resolve/main/TSP/tsp100_am.ckpt?download=true',
}


class TSPAM(REINFORCE):
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
        num_nodes: int=50,
        # network
        embedding_dim: int=128,
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
        # baseline
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs={},
        # local_search
        local_search_type: str=None,
        **kwargs
    ):
        self.num_nodes = num_nodes
        env = TSPEnv(num_loc=num_nodes)
        
        encoder = GraphAttentionEncoder(
            env_name=env.name,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )

        decoder = AutoregressiveDecoder(
            env_name=env.name,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        
        policy = AttentionModelPolicy(
            env_name=env.name,
            embedding_dim=embedding_dim,
            encoder=encoder, 
            decoder=decoder, 
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            decoding_type=decoding_type
        )
        
        super(TSPAM, self).__init__(
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
            baseline=baseline,
            baseline_kwargs=baseline_kwargs,
            local_search_type=local_search_type,
            **kwargs
        )

        self.setup()

    def load_ckpt(self, ckpt_path:str=None, load_baseline: bool=True):
        # load state_dict from pretrained checkpoint
        if ckpt_path is None:
            if self.num_nodes in [50, 100]:
                url = tsp_am_path[self.num_nodes]
                filename=f"ckpts/tsp{self.num_nodes}_am.ckpt"
                pygm.utils.download(filename=filename, url=url, to_cache=None)
                state_dict = torch.load(filename)['state_dict']
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