from typing import Union

import torch.nn as nn

from tensordict import TensorDict

from models.rl4co.nn.utils import get_log_likelihood
from models.rl4co.common.decoder import AutoregressiveDecoder
from models.rl4co.common.encoder import GraphAttentionEncoder
from models.rl4co.envs import RL4COEnvBase, get_env
from models.rl4co.utils import get_pylogger

log = get_pylogger(__name__)


class AutoregressivePolicy(nn.Module):
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
        env_name: str,
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
        decoding_type: str="greedy",
        **unused_kw,
    ):
        super(AutoregressivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        self.env_name = env_name

        if encoder is None:
            log.info("Initializing default GraphAttentionEncoder")
            self.encoder = GraphAttentionEncoder(
                env_name=self.env_name,
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
            log.info("Initializing default AutoregressiveDecoder")
            self.decoder = AutoregressiveDecoder(
                env_name=self.env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
                mask_inner=mask_inner,
                force_flash_attn=force_flash_attn,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
            )
        else:
            self.decoder = decoder
        
        self.train_decode_type = decoding_type if train_decode_type is None else train_decode_type
        self.val_decode_type = decoding_type if val_decode_type is None else val_decode_type
        self.test_decode_type = decoding_type if test_decode_type is None else test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
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

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # DECODER: main rollout with autoregressive decoding
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
