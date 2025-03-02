from argparse import ArgumentParser


def generate_arg_parser():
    parser = ArgumentParser(description='Arguments for generating tsp data.')
    
    parser.add_argument("--type", type=str, choices=["uniform", "cluster", "cluster_fixed_centers", "gaussian", "tsplibs"], default="uniform")
    
    if parser.parse_known_args()[0].type == "tsplibs":
        parser.add_argument("--mode", type=str, choices=["read", "calculate"], default="read")
        if parser.parse_known_args()[0].mode == "read":
            parser.add_argument("--tsplibs_path", type=str, default="data/tsp/tsplibs/raw/read")
        else:
            parser.add_argument("--tsplibs_path", type=str, default="data/tsp/tsplibs/raw/calculate")
        
    else:
        parser.add_argument("--min_nodes", type=int, default=50)
        parser.add_argument("--max_nodes", type=int, default=50)
        parser.add_argument("--num_samples", type=int, default=130560)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--filename", type=str, default=None)
        parser.add_argument("--train_filename", type=str, default=None)
        parser.add_argument("--valid_filename", type=str, default=None)
        parser.add_argument("--test_filename", type=str, default=None)
        parser.add_argument("--regret_dir", type=str, default=None)
        parser.add_argument("--ratio", type=str, default="128000:1280:1280")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--calc_regret", action='store_true')
        parser.add_argument("--solver", type=str, default="lkh")
        
        if parser.parse_known_args()[0].type == "cluster":
            parser.add_argument("--num_clusters", type=int, default=10)
            parser.add_argument("--cluster_std", type=float, default=0.1)
    
        if parser.parse_known_args()[0].type == "gaussian":
            parser.add_argument("--mean_x", type=float, default=0.0)
            parser.add_argument("--mean_y", type=float, default=0.0)
            parser.add_argument("--gaussian_std", type=float, default=1)

    args = parser.parse_args()
    return args  


def nar_train_arg_parser():
    parser = ArgumentParser(description='Arguments and hyperparameters for training learning-driven solvers for CO.')
    
    # define the model
    parser.add_argument('--task', choices=['tsp', 'mis'], type=str, required=True)
    parser.add_argument('--encoder', choices=['gnn', 'gnn-wise', 'us', 'diffusion', 'gnn4reg', 'dimes'], type=str, required=True)
    
    # num_nodes is needed 
    parser.add_argument('--num_nodes', type=int, required=True)
    
    # network args
    parser.add_argument('--network_type', type=str, choices=['gnn', 'gat', 'sag'], default='gnn')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_channels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--sparse_factor', type=int, default=-1)

    # utsp
    parser.add_argument('--temperature', type=float, default=3.5)
    parser.add_argument('--distance_loss', type=float, default=1.0)
    parser.add_argument('--row_wise_loss', type=float, default=10.0)
    parser.add_argument('--loop_loss', type=float, default=0.1)

    # diffusion args
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=50)
    parser.add_argument('--inference_schedule', type=str, default='cosine')
    parser.add_argument('--inference_trick', type=str, default="ddim")

    # DIMES args
    parser.add_argument('--inner_epochs', type=int, default=100)
    parser.add_argument('--inner_samples', type=int, default=2000)
    parser.add_argument('--inner_lr', type=float, default=5e-2)

    # batch size and files
    N = parser.parse_known_args()[0].num_nodes
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--train_file', type=str, default=f'data/tsp/uniform/train/tsp{N}_uniform_1.28m.txt')
    parser.add_argument('--valid_file', type=str, default=f'data/tsp/uniform/valid/tsp{N}_uniform_val.txt')
    parser.add_argument('--regret_dir', type=str, default=None)
    parser.add_argument('--valid_samples', type=int, default=1280)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=-1) # for DIMES

    # parallel/active
    parser.add_argument('--parallel_sampling', type=int, default=1)
    parser.add_argument('--active_search', action='store_true', help='enable active search')
    parser.add_argument('--as_steps', type=int, default=100)
    parser.add_argument('--as_samples', type=int, default=1000)
    
    # precision
    parser.add_argument('--fp16', action='store_true')
    
    # logger args
    parser.add_argument("--wandb_logger_name", type=str, default="wandb")
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    
    # checkpoint args
    parser.add_argument("--monitor", type=str, default="val/loss", help="checkpoint's monitor")
    parser.add_argument("--every_n_epochs", type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=50) # for DIMES
    parser.add_argument("--every_n_train_steps", type=int, default=None) # for DIMES
    parser.add_argument("--val_check_interval", type=int, default=None) # for DIMES
    
    # learning
    # parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='cosine-decay')
    parser.add_argument('--lr_decay', type=float, default='0.99')
    
    args = parser.parse_args()
    args.mode = "train"

    return args


def ar_train_arg_parser():
    parser = ArgumentParser(description='Arguments and hyperparameters for training learning-driven solvers for CO.')
    
    # define the model
    parser.add_argument('--task', choices=['tsp', 'mis'], type=str, required=True)
    parser.add_argument('--encoder', choices=['am', 'pomo', 'symnco'], type=str, required=True)
    
    # num_nodes is needed 
    parser.add_argument('--num_nodes', type=int, required=True)
    
    # network args
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)

    # train/valid files
    N = parser.parse_known_args()[0].num_nodes    
    parser.add_argument('--train_batch_size', type=int, default=640)
    parser.add_argument('--valid_batch_size', type=int, default=640)
    parser.add_argument('--train_data_size', type=int, default=1280000)
    parser.add_argument('--valid_file', type=str, default=f'data/tsp/uniform/valid/tsp{N}_uniform_val.txt')
    parser.add_argument('--train_decode_type', type=str, default="sampling")
    parser.add_argument('--val_decode_type', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)

    # precision
    parser.add_argument('--fp16', action='store_true')
    
    # logger args
    parser.add_argument("--wandb_logger_name", type=str, default="wandb")
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    
    # checkpoint args
    parser.add_argument("--monitor", type=str, default="val/reward", help="checkpoint's monitor")
    parser.add_argument("--every_n_epochs", type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    # learning
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--lr_decay', type=float, default='0.99')

    args = parser.parse_args()
    args.mode = "train"

    return args


def nar_test_arg_parser():
    parser = ArgumentParser(description='Arguments and hyperparameters for training learning-driven solvers for CO.')
    
    # define the model
    parser.add_argument('--task', choices=['tsp', 'mis'], type=str, required=True)
    parser.add_argument('--encoder', choices=['gnn', 'gnn-wise', 'us', 'diffusion', 'gnn4reg', 'dimes'], type=str, required=True)
    
    # num_nodes and mode is needed 
    parser.add_argument('--num_nodes', type=int, required=True)
    
    # network args
    parser.add_argument('--network_type', type=str, choices=['gnn', 'gat', 'sag'], default='gnn')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_channels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    
    # utsp
    parser.add_argument('--temperature', type=float, default=3.5)
    parser.add_argument('--distance_loss', type=float, default=1.0)
    parser.add_argument('--row_wise_loss', type=float, default=10.0)
    parser.add_argument('--loop_loss', type=float, default=0.1)

    # diffusion args
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=50)
    parser.add_argument('--inference_schedule', type=str, default='cosine')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--gradient_search', action='store_true')
    parser.add_argument('--rewrite_ratio', type=float, default=0.4)
    parser.add_argument('--rewrite_steps', type=int, default=3)
    parser.add_argument('--steps_inf', type=int, default=5)
    

    # batch size and files
    N = parser.parse_known_args()[0].num_nodes
    parser.add_argument('--test_file', type=str, default=f'data/tsp/uniform/test/tsp{N}_uniform_test.txt')
    parser.add_argument('--regret_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # parallel/active
    parser.add_argument('--parallel_sampling', type=int, default=1)
    parser.add_argument('--active_search', action='store_true', help='enable active search')
    parser.add_argument('--as_steps', type=int, default=100)
    parser.add_argument('--as_samples', type=int, default=1000)
    parser.add_argument('--inner_lr', type=float, default=5e-2)
    
    # precision
    parser.add_argument('--fp16', action='store_true')
    
    # logger args
    parser.add_argument("--wandb_logger_name", type=str, default="wandb")
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    
    # checkpoint args
    parser.add_argument("--monitor", type=str, default="test/loss", help="checkpoint's monitor")
    parser.add_argument("--every_n_epochs", type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    # decoding
    parser.add_argument("--decoding_type", type=str, choices=["greedy", "beam", "rg_mcts", \
        'mcts_solver', 'beam_mcts', 'rmcts'], required=False, default="greedy")

    # 2opt
    parser.add_argument("--max_iterations_2opt", type=int, required=False, default=5000)

    # beam search
    parser.add_argument('--beam_size', type=int, default=1280)
    parser.add_argument('--beam_random_smart', action='store_true')
    parser.add_argument('--beam_probs_type', choices=['raw', 'logits'], type=str, default='raw')

    # random_greedy_mcts
    parser.add_argument('--random_samples', type=int, default=10)
    parser.add_argument('--random_weight', type=float, default=3.0)

    # mcts
    parser.add_argument('--mcts_max_depth', type=int, default=10)
    parser.add_argument('--mcts_max_iterations_2opt', type=int, default=None)
    parser.add_argument('--mcts_time_limit', type=float, default=None)
    parser.add_argument('--mcts_smooth', action='store_true')
    parser.add_argument('--mcts_smooth_v2', action='store_true')
    
    # local_search
    parser.add_argument('--local_search_type', choices=['2opt', 'relocate', 'ls', "gls", "mcts"], type=str, default=None)

    # guided local search
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=100)

    args = parser.parse_args()
    args.mode = "test"
    
    return args


def ar_test_arg_parser():
    parser = ArgumentParser(description='Arguments and hyperparameters for training learning-driven solvers for CO.')
    
    # define the model
    parser.add_argument('--task', choices=['tsp', 'mis'], type=str, required=True)
    parser.add_argument('--encoder', choices=['am', 'pomo', 'symnco'], type=str, required=True)
    
    # num_nodes and mode is needed 
    parser.add_argument('--num_nodes', type=int, required=True)
    
    # network args
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)

    # test file
    N = parser.parse_known_args()[0].num_nodes    
    parser.add_argument('--test_file', type=str, default=f'data/tsp/uniform/test/tsp{N}_uniform_test.txt')
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)

    # precision
    parser.add_argument('--fp16', action='store_true')
    
    # logger args
    parser.add_argument("--wandb_logger_name", type=str, default="wandb")
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    
    # checkpoint args
    parser.add_argument("--monitor", type=str, default="test/reward", help="checkpoint's monitor")
    parser.add_argument("--every_n_epochs", type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # EAS
    parser.add_argument("--eas_type", type=str, default=None)
    parser.add_argument("--eas_batch_size", type=int, default=2)
    parser.add_argument("--eas_max_iters", type=int, default=200)
    parser.add_argument("--eas_filename", type=str, default=None)
    parser.add_argument("--eas_max_epochs", type=int, default=1)
    parser.add_argument('--eas_save_file', action='store_true')
    
    # decoding
    parser.add_argument("--decoding_type", type=str, choices=["greedy", "sampling", \
      "multistart_sampling", "multistart_greedy"], required=False, default="greedy")
    
    # 2opt
    parser.add_argument("--max_iterations_2opt", type=int, required=False, default=5000)

    # local_search
    parser.add_argument('--local_search_type', choices=['2opt'], type=str, default=None)
    
    args = parser.parse_args()
    args.mode = "test"

    return args