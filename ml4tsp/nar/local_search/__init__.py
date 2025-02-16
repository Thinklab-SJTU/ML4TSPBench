from .base import ML4TSPNARLocalSearch
from .mcts import ML4TSPNARMCTS
from .two_opt import ML4TSPNARTwoOpt
from .guided_ls import ML4TSPNARGuidedLS


def get_local_search_by_name(name: str):
    local_search_dict = {
        "mcts":  ML4TSPNARMCTS,
        "2opt": ML4TSPNARTwoOpt,
    }
    supported_local_search_type = local_search_dict.keys()
    if name not in supported_local_search_type:
        message = (
            f"Cannot find the local_search whose name is {name},"
            f"ML4TSPNARLocalSearch only supports {supported_local_search_type}."
        )
        raise ValueError(message)
    return local_search_dict[name]