from .envs_base import RL4COEnvBase

from .atsp_env import ATSPEnv
from .cvrp_env import CVRPEnv
from .dpp_env import DPPEnv
from .mdpp_env import MDPPEnv
from .mtsp_env import MTSPEnv
from .op_env import OPEnv
from .pctsp_env import PCTSPEnv
from .pdp_env import PDPEnv
from .sdvrp_env import SDVRPEnv
from .spctsp_env import SPCTSPEnv
from .tsp_env import TSPEnv

from .envs_utils import batch_to_scalar, make_composite_from_td

ENV_REGISTRY = {
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "dpp": DPPEnv,
    "mdpp": MDPPEnv,
    "mtsp": MTSPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "pdp": PDPEnv,
    "sdvrp": SDVRPEnv,
    "spctsp": SPCTSPEnv,
    "tsp": TSPEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)