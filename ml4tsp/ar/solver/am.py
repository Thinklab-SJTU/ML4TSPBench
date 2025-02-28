from ml4tsp.ar.model.am import ML4TSPAM
from ml4tsp.ar.solver.base import ML4TSPARSolver


class ML4TSPAMSolver(ML4TSPARSolver):
    def __init__(self, model: ML4TSPAM):
        super(ML4TSPAMSolver, self).__init__(model)