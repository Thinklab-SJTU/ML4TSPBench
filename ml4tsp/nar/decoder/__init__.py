from .base import ML4TSPNARDecoder
from .greedy import ML4TSPNARGreeyDecoder
from .beam import ML4TSPNARBeamDecoder
from .sampling import ML4TSPNARSamplingDecoder
from .random import ML4TSPNARRandomDecoder
from .insertion import ML4TSPNARInsertionDecoder
from .mcts import ML4TSPNARMCTSDecoder


def get_nar_decoder_by_name(name: str):
    decoder_dict = {
        "greedy": ML4TSPNARGreeyDecoder,
        "beam": ML4TSPNARBeamDecoder,
        "sampling": ML4TSPNARSamplingDecoder,
        "random": ML4TSPNARRandomDecoder,
        "insertion": ML4TSPNARInsertionDecoder,
        "mcts": ML4TSPNARMCTSDecoder
    }
    supported_decoder_type = decoder_dict.keys()
    if name not in supported_decoder_type:
        message = (
            f"Cannot find the decoder whose name is ``{name}``, "
            f"ML4TSPNARDecoder only supports {supported_decoder_type}."
        )
        raise ValueError(message)
    return decoder_dict[name]