import torch
from typing import Union
from tensordict import TensorDict


def gather_by_index(
    src: torch.Tensor, idx: torch.Tensor, dim: int = 1, squeeze: bool = True
) -> torch.Tensor:
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)


def get_num_starts(td: TensorDict) -> torch.Tensor:
    """Returns the number of possible start nodes for the environment based on the action mask"""
    return td["action_mask"].shape[-1]


def select_start_nodes(td: TensorDict, nodes_num: int = None):
    """Node selection strategy as proposed in POMO (Kwon et al. 2020)
    and extended in SymNCO (Kim et al. 2022).
    Selects different start nodes for each batch element

    Args:
        td: TensorDict containing the data. We may need to access the available actions to select the start nodes
        nodes_num: Number of nodes to select
    """
    nodes_num = get_num_starts(td) if nodes_num is None else nodes_num

    # Environments with depot: don't select the depot as start node
    selected = torch.arange(nodes_num, device=td.device).repeat_interleave(td.shape[0])
    return selected


def _batchify_single(
    x: Union[torch.Tensor, TensorDict], repeats: int
) -> Union[torch.Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[torch.Tensor, TensorDict], shape: torch.Union[tuple, int]
) -> Union[torch.Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(
    x: Union[torch.Tensor, TensorDict], repeats: int
) -> Union[torch.Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[torch.Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[torch.Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def get_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)


def get_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor:
    """Compute the total tour distance for a batch of ordered tours.
    Computes the L2 norm between each pair of consecutive nodes in the tour and sums them up.

    Args:
        ordered_locs: Tensor of shape [batch_size, num_nodes, 2] containing the ordered locations of the tour
    """
    ordered_locs_next = torch.roll(ordered_locs, 1, dims=-2)
    return get_distance(ordered_locs_next, ordered_locs).sum(-1)