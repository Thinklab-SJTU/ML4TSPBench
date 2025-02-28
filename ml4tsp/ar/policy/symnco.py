import torch
from torch import nn
from einops import rearrange
from torchrl.modules.models import MLP
from tensordict.tensordict import TensorDict
from torch.nn.functional import cosine_similarity
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.policy.base import ML4TSPARPolicy


class ML4TSPSymNCOPolicy(ML4TSPARPolicy):
    """SymNCO Policy based on AutoregressivePolicy.
    This differs from the default :class:`AutoregressivePolicy` in that it
    projects the initial embeddings to a lower dimension using a projection head and
    returns it. This is used in the SymNCO algorithm to compute the invariance loss.
    Based on Kim et al. (2022) https://arxiv.org/abs/2205.13209.

    Args:
        env_name: Name of the environment
        embedding_dim: Dimension of the embedding
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the encoder
        normalization: Normalization to use in the encoder
        projection_head: Projection head to use
        use_projection_head: Whether to use projection head
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        projection_head: nn.Module = None,
        use_projection_head: bool = True,
        **kwargs,
    ):
        super(ML4TSPSymNCOPolicy, self).__init__(
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )

        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            self.projection_head = (
                MLP(embedding_dim, embedding_dim, 1, embedding_dim, nn.ReLU)
                if projection_head is None
                else projection_head
            )

    def forward(
        self,
        td: TensorDict,
        env: ML4TSPAREnv,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = True,
        **decoder_kwargs,
    ) -> dict:
        super().forward.__doc__  # trick to get docs from parent class

        # Ensure that if use_projection_head is True, then return_init_embeds is True
        assert not (
            self.use_projection_head and not return_init_embeds
        ), "If `use_projection_head` is True, then we must `return_init_embeds`"

        out = super().forward(
            td,
            env,
            phase,
            return_actions,
            return_entropy,
            return_init_embeds,
            **decoder_kwargs,
        )

        if phase == 'test':
            return out

        # Project initial embeddings
        if self.use_projection_head:
            out["proj_embeddings"] = self.projection_head(out["init_embeds"])

        return out


def problem_symmetricity_loss(
    reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """REINFORCE loss for problem symmetricity
    Baseline is the average reward for all augmented problems
    Corresponds to `L_ps` in the SymNCO paper
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def solution_symmetricity_loss(
    reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """REINFORCE loss for solution symmetricity
    Baseline is the average reward for all start nodes
    Corresponds to `L_ss` in the SymNCO paper
    """
    num_starts = reward.shape[dim]
    if num_starts < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def invariance_loss(
    proj_embed: torch.Tensor, num_augment: int
) -> torch.Tensor:
    """Loss for invariant representation on projected nodes
    Corresponds to `L_inv` in the SymNCO paper
    """
    pe = rearrange(proj_embed, "(b a) ... -> b a ...", a=num_augment)
    similarity = sum(
        [cosine_similarity(pe[:, 0], pe[:, i], dim=-1) for i in range(1, num_augment)]
    )
    return similarity.mean()
