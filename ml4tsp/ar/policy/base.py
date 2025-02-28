import torch
from torch import nn
from tensordict import TensorDict
from pytorch_lightning.utilities import rank_zero_info
from ml4tsp.ar.env.ar_env import ML4TSPAREnv
from ml4tsp.ar.encoder.gat.gat_encoder import GATEncoder
from ml4tsp.ar.decoder.decoder import ML4TSPARDecoder


class ML4TSPARPolicy(nn.Module):
    """Base Auto-regressive policy for NCO construction methods.
    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (autoregressively) to construct the solution to the NCO problem
    Based on the policy from Kool et al. (2019) and extended for common use on multiple models in RL4CO.

    Note:
        We recommend to provide the decoding method as a keyword argument to the
        decoder during actual testing. The `{phase}_decode_type` arguments are only
        meant to be used during the main training loop. You may have a look at the
        evaluation scripts for examples.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes.
        decoder: Decoder module. Can be passed by sub-classes.
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        context_embedding: Model to use for the context embedding. If None, use the default embedding for the environment
        dynamic_embedding: Model to use for the dynamic embedding. If None, use the default embedding for the environment
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        mask_inner: Whether to mask the inner diagonal in the attention layers
        use_graph_context: Whether to use the initial graph context to modify the query
        force_flash_attn: Whether to force the use of flash attention in the attention layers
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **unused_kw: Unused keyword arguments
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        mask_inner: bool = True,
        use_graph_context: bool = True,
        force_flash_attn: bool = False,
        train_decode_type: str = None,
        val_decode_type: str = None,
        test_decode_type: str = None,
        decode_type: str="greedy",
        **unused_kw,
    ):
        super(ML4TSPARPolicy, self).__init__()

        if len(unused_kw) > 0:
            rank_zero_info(f"Unused kwargs: {unused_kw}")

        if encoder is None:
            rank_zero_info("Initializing default GATEncoder")
            self.encoder = GATEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                force_flash_attn=force_flash_attn,
                init_embedding=init_embedding,
            )
        else:
            self.encoder = encoder

        if decoder is None:
            rank_zero_info("Initializing default AutoregressiveDecoder")
            self.decoder = ML4TSPARDecoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
                mask_inner=mask_inner,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
            )
        else:
            self.decoder = decoder
        
        self.train_decode_type = decode_type if train_decode_type is None else train_decode_type
        self.val_decode_type = decode_type if val_decode_type is None else val_decode_type
        self.test_decode_type = decode_type if test_decode_type is None else test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: ML4TSPAREnv,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        **decoder_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            decoder_kwargs: Keyword arguments for the decoder

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        embeddings, init_embeds = self.encoder(td)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        log_p, actions, td_out = self.decoder(td, embeddings, env, **decoder_kwargs)
        if phase == 'test':
            return log_p, actions, td_out
        
        # Log likelihood is calculated within the model
        log_likelihood = get_log_likelihood(log_p, actions, td_out.get("mask", None))

        out = {
            "reward": td_out["reward"],
            "log_likelihood": log_likelihood,
        }
        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(log_p.exp() * log_p).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        if return_init_embeds:
            out["init_embeds"] = init_embeds

        return out
    
    def __repr__(self):
        return f"{self.__class__.__name__}"


def get_log_likelihood(
    log_p: torch.Tensor, 
    actions: torch.Tensor,
    mask: torch.Tensor, 
    return_sum: bool = True
) -> torch.Tensor:
    """Get log likelihood of selected actions"""

    log_p = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[~mask] = 0

    assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return log_p.sum(1)  # [batch]
    else:
        return log_p  # [batch, decode_len]