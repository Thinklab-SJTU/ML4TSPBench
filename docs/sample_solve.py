import os
import torch
import numpy as np
from solvers import TSPNARSolver
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if torch.cuda.is_available():
    device = 'cuda'
    print(f"CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    device = 'cpu'
    print("CUDA is not available. Using CPU.")

solver = TSPNARSolver()
solver.from_txt("data/uniform/test/tsp500_lkh_50000_16.54811.txt")
tours = solver.solve(
        batch_size=1,
        sparse_factor=50,
        encoder='diffusion',
        encoder_kwargs={
            # "ckpt_path": "",
            # "regret_dir": ""
        },
        decoding_type="greedy", 
        local_search_type="2opt",
        decoding_kwargs={
            "beam_probs_type": 'logits',
            "beam_size": 10,
            "beam_random_smart": False,
            "random_samples": 10,
            "mcts_max_depth": 10,
            "mcts_param_t": 0.1,
        },
        ls_kwargs={
            "mcts_smooth": False,
            "time_limit": 0.5,
            "perturbation_moves": 100,
        },
        device='cuda',
    )
return_tuple = solver.evaluate(caculate_gap=True)
print(return_tuple)