from .gnn import MetaGNN, TSPGNN, TSPGNNWISE, UTSPGNN, TSPGNN4REG, TSPDIMES
from .rl import TSPAM, TSPPOMO, TSPSymNCO
from .diffusion import TSPDiffusion
from .base import TSPNARTrainer, TSPNAREncoder, Checkpoint, Logger, TSPAREncoder, TSPARTrainer


NAR_MODEL = {
    ("tsp", "gnn"): TSPGNN,
    ("tsp","gnn-wise"): TSPGNNWISE,
    ("tsp","us"): UTSPGNN,
    ("tsp", "gnn4reg"): TSPGNN4REG,
    ("tsp", "dimes"): TSPDIMES,
    ("tsp", "diffusion"): TSPDiffusion
}


AR_MODEL = {
    ("tsp", "am"): TSPAM,
    ("tsp", "pomo"): TSPPOMO,
    ("tsp", "symnco"): TSPSymNCO    
}


def get_nar_model(task="tsp", name="gnn"):
    return NAR_MODEL[(task, name)]


def get_ar_model(task="tsp", name="gnn"):
    return AR_MODEL[(task, name)]
