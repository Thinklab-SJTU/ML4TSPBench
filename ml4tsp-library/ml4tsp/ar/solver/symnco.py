from ml4tsp.ar.model.symnco import ML4TSPSymNCO
from ml4tsp.ar.solver.base import ML4TSPARSolver


class ML4TSPSymNCOSolver(ML4TSPARSolver):
    def __init__(self, model: ML4TSPSymNCO):
        super(ML4TSPSymNCOSolver, self).__init__(model)