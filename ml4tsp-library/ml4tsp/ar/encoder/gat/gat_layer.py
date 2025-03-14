import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    print(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention"
    )

    def scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
        if scale is None:
            scale = math.sqrt(Q.size(-1))  # scale factor
        # compute the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # apply causal masking if required
        if is_causal:
            mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        # compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        # apply dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        # compute the weighted sum of values
        return torch.matmul(attn_probs, V)


def flash_attn_wrapper(self, func, *args, **kwargs):
    """Wrapper for flash attention to automatically cast to fp16 if needed"""
    if self.force_flash_attn and args[0].is_cuda:
        original_dtype = args[0].dtype
        args = [arg.half() for arg in args if isinstance(arg, torch.Tensor)]
        out = func(*args, **kwargs)
        return out.to(original_dtype)
    else:
        return func(*args, **kwargs)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    

class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.
        In case you want to use Flash Attention, you may have a look at the MHA module under `rl4co.bench4co.nn.flash_attention.MHA`.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        force_flash_attn: whether to force flash attention. If True, then we automatically cast to fp16
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device=None,
        dtype=None,
        force_flash_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.force_flash_attn = force_flash_attn
        self.attention_dropout = attention_dropout

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        # Scaled dot product attention
        out = self.flash_attn_wrapper(
            scaled_dot_product_attention,
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

    flash_attn_wrapper = flash_attn_wrapper


class LogitAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.

    Note:
        With Flash Attention, masking is not supported

    Perform the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key
        4. Normalize and mask

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        tanh_clipping: tanh clipping value
        mask_inner: whether to mask inner attention
        mask_logits: whether to mask logits
        normalize: whether to normalize logits
        softmax_temp: softmax temperature
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalize: bool = True,
        softmax_temp: float = 1.0,
        linear_bias: bool = False,
        sdp_fn=scaled_dot_product_attention,
    ):
        super(LogitAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_logits = mask_logits
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.normalize = normalize
        self.softmax_temp = softmax_temp

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.sdp_fn = sdp_fn

    def forward(self, query, key, value, logit_key, mask, softmax_temp=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, mask)
        glimpse = self.project_out(heads)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (
            torch.bmm(glimpse, logit_key.squeeze(1).transpose(-2, -1))
            / math.sqrt(glimpse.size(-1))
        ).squeeze(1)

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits:
            logits[mask] = float("-inf")

        # Normalize with softmax and apply temperature
        if self.normalize:
            softmax_temp = softmax_temp if softmax_temp is not None else self.softmax_temp
            logits = torch.log_softmax(logits / softmax_temp, dim=-1)

        assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)

        if self.mask_inner:
            # need to invert mask: (N L S) -> (N 1 L S)
            attn_mask = (
                ~mask.unsqueeze(1) if mask.ndim == 3 else ~mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None

        heads = self.sdp_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)


class MultiHeadAttentionLayer(nn.Sequential):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        force_flash_attn: whether to force FlashAttention (move to half precision)
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        force_flash_attn: bool = False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    embed_dim, num_heads, force_flash_attn=force_flash_attn
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embedding_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        force_flash_attn: whether to force FlashAttention (move to half precision)
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        force_flash_attn: bool = False,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                    force_flash_attn=force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x)
        return h