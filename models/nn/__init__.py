from .gnn import GNNEncoder
from .gat import GATEncoder
from .sag import SAGEncoder

NET_CLASS = {
    'gnn': GNNEncoder,
    'gat': GATEncoder,
    'sag': SAGEncoder
}