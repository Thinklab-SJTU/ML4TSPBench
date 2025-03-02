import torch
from torch import nn
from tensordict import TensorDict
from ml4tsp.ar.utils.ops import gather_by_index


class TSPInitEmbedder(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(TSPInitEmbedder, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out
    

class EnvContextEmbedder(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(
        self, 
        embedding_dim: int, 
        step_context_dim: int = None, 
        linear_bias: bool = False
    ):
        super(EnvContextEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        step_context_dim = (
            step_context_dim if step_context_dim is not None else embedding_dim
        )
        self.project_context = nn.Linear(
            step_context_dim, embedding_dim, bias=linear_bias
        )

    def _cur_node_embedding(self, embeddings: torch.Tensor, td: TensorDict):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings: torch.Tensor, td: TensorDict):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings: torch.Tensor, td: TensorDict):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class TSPContextEmbedder(EnvContextEmbedder):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embedding_dim: int):
        super(TSPContextEmbedder, self).__init__(embedding_dim, 2 * embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings: torch.Tensor, td: TensorDict):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)
    

class StaticEmbedder(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedder, self).__init__()

    def forward(self, td: TensorDict):
        return 0, 0, 0