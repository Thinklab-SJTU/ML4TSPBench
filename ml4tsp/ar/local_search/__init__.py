from .base import ML4TSPARLocalSearch
from .two_opt import ML4TSPARTwoOpt


def get_local_search_by_name(name: str):
    local_search_dict = {
        "2opt": ML4TSPARTwoOpt,
    }
    supported_local_search_type = local_search_dict.keys()
    if name not in supported_local_search_type:
        message = (
            f"Cannot find the local_search whose name is {name},"
            f"ML4TSPNARLocalSearch only supports {supported_local_search_type}."
        )
        raise ValueError(message)
    return local_search_dict[name]