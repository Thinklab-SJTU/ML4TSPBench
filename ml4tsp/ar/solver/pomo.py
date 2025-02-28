from ml4tsp.ar.model.pomo import ML4TSPPOMO
from ml4tsp.ar.solver.base import ML4TSPARSolver


class ML4TSPPOMOSolver(ML4TSPARSolver):
    def __init__(self, model: ML4TSPPOMO):
        super(ML4TSPPOMOSolver, self).__init__(model)