import math
import functools
import torch
import torch.utils.checkpoint as activation_checkpoint
from torch import nn
from torch_sparse import SparseTensor
from .gnn_layer import GNNLayer
from .gnn_embedders import (
    ScalarEmbeddingSine1D, ScalarEmbeddingSine3D, PositionEmbeddingSine
)


class GNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    def __init__(
        self, 
        num_layers: int = 12, 
        hidden_dim: int = 256, 
        output_channels: int = 2, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False, 
        gated: bool = True,
        sparse: bool = False, 
        use_activation_checkpoint: bool = False, 
        node_feature_only: bool = False, 
        time_embed_flag: bool = False,
    ):
        super(GNNEncoder, self).__init__()
        self.node_feature_only = node_feature_only
        self.sparse = sparse
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        time_embed_dim = hidden_dim // 2
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        self.time_embed_flag = time_embed_flag

        if not node_feature_only:
            self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
            self.edge_pos_embed = ScalarEmbeddingSine3D(hidden_dim, normalize=False)
        else:
            self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)

        if self.time_embed_flag:
            self.time_embed = nn.Sequential(
                nn.Linear(hidden_dim, time_embed_dim),
                nn.ReLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.time_embed_layers = nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(
                        time_embed_dim,
                        hidden_dim,
                    ),
                ) for _ in range(num_layers)
            ])

        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            # zero_module(
                nn.Conv2d(hidden_dim, output_channels, kernel_size=1, bias=True)
            # ),
        )
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(num_layers)
        ])

        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
        self.use_activation_checkpoint = use_activation_checkpoint
        
    def dense_forward(
        self, 
        x: torch.Tensor, 
        graph: torch.Tensor, 
        edge_index: torch.Tensor, 
        timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input node coordinates (B x V x 2)
            graph: Graph adjacency matrices (B x V x V)
            edge_index: Edge indices (2 x E)
            timesteps: Input node timesteps (B)
        Returns:
            Updated edge features (B x V x V)
        """
        # Embed edge features
        del edge_index
        x = self.node_embed(self.pos_embed(x)) 
        e = self.edge_embed(self.edge_pos_embed(graph))
        
        if self.time_embed_flag:
            time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
        graph = torch.ones_like(graph).long()
        if self.time_embed_flag:
            for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
                x_in, e_in = x, e
                x, e = layer(x, e, graph, mode="direct")
                if not self.node_feature_only:
                    e = e + time_layer(time_emb)[:, None, None, :]
                else:
                    x = x + time_layer(time_emb)[:, None, :]
                x = x_in + x
                e = e_in + out_layer(e)
        else:
            for layer, out_layer in zip(self.layers, self.per_layer_out):
                x_in, e_in = x, e
                x, e = layer(x, e, graph, mode="direct")
                x = x_in + x
                e = e_in + out_layer(e)
        e = self.out(e.permute((0, 3, 1, 2)))
        return e

    def sparse_forward(
        self, 
        x: torch.Tensor, 
        graph: torch.Tensor, 
        edge_index: torch.Tensor, 
        timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input node coordinates (B x V x 2)
            graph: Graph edge features (E)
            edge_index: Adjacency matrix for the graph (2 x E)
            timesteps: Input edge timestep features (E)
        Returns:
            Updated edge features (E x H)
        """
        # Embed edge features
        x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
        e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)) if self.time_embed_flag else None
        edge_index = edge_index.long()
        
        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
        e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))
        return e
    
    def sparse_forward_node_feature_only(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timesteps: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.node_embed(self.pos_embed(x))
        x_shape = x.shape
        e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)) if self.time_embed_flag else None
        edge_index = edge_index.long()

        x, e = self.sparse_encoding(x, e, edge_index, time_emb)
        x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
        x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
        return x
    
    def sparse_encoding(
        self, 
        x: torch.Tensor, 
        e: torch.Tensor, 
        edge_index: torch.Tensor, 
        time_emb: torch.Tensor = None
    ):
        adj_matrix = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=torch.ones_like(edge_index[0].float()),
            sparse_sizes=(x.shape[0], x.shape[0]),
        )
        adj_matrix = adj_matrix.to(x.device)

        if self.time_embed_flag:
            for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
                x_in, e_in = x, e

                if self.use_activation_checkpoint:
                    single_time_emb = time_emb[:1]

                    run_sparse_layer_fn = functools.partial(
                        run_sparse_layer,
                        add_time_on_edge=not self.node_feature_only
                    )

                    out = activation_checkpoint.checkpoint(
                        run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
                        x_in, e_in, single_time_emb
                    )
                    x = out[0]
                    e = out[1]
                else:
                    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
                    if not self.node_feature_only:
                        e = e + time_layer(time_emb)
                    else:
                        x = x + time_layer(time_emb)
                    x = x_in + x
                    e = e_in + out_layer(e)
        else:
            for layer, out_layer in zip(self.layers, self.per_layer_out):
                x_in, e_in = x, e

                if self.use_activation_checkpoint:
                    run_sparse_layer_fn = functools.partial(
                        run_sparse_layer,
                        add_time_on_edge=not self.node_feature_only
                    )

                    out = activation_checkpoint.checkpoint(
                        run_sparse_layer_fn(layer, None, out_layer, adj_matrix, edge_index),
                        x_in, e_in, None
                    )
                    x = out[0]
                    e = out[1]
                else:
                    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
                    x = x_in + x
                    e = e_in + out_layer(e)
        return x, e

    def forward(
        self, 
        x: torch.Tensor, 
        graph: torch.Tensor = None,
        edge_index: torch.Tensor = None, 
        timesteps: torch.Tensor = None
    ):
        if self.node_feature_only:
            if self.sparse:
                return self.sparse_forward_node_feature_only(x, edge_index, timesteps)
            else:
                raise NotImplementedError
        else:
            if self.sparse:
                return self.sparse_forward(x, graph, edge_index, timesteps)
            else:
                return self.dense_forward(x, graph, edge_index, timesteps)
            
            
def run_sparse_layer(
    layer: nn.Module,
    time_layer: nn.Module,
    out_layer: nn.Module, 
    adj_matrix: torch.Tensor, 
    edge_index: torch.Tensor, 
    add_time_on_edge: bool = True
):
    def custom_forward(*inputs):
        x_in = inputs[0]
        e_in = inputs[1]
        time_emb = inputs[2]
        x, e = layer(
            x_in, e_in, adj_matrix, mode="direct", 
            edge_index=edge_index, sparse=True
        )
        if not (time_layer is None):
            if add_time_on_edge:
                e = e + time_layer(time_emb)
            else:
                x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
        return x, e
    return custom_forward


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

    
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
     

class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type(x.dtype)

  
def normalization(channels: int):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module