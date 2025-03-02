import time
import numpy as np
from tqdm import tqdm
from typing import Union
from ml4co_kit import TSPSolver, to_tensor, SOLVER_TYPE, TSPEvaluator, to_numpy
from ml4tsp.ar.model.base import ML4TSPARBaseModel


class ML4TSPARSolver(TSPSolver):
    def __init__(self, model: ML4TSPARBaseModel):
        super(ML4TSPARSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP, scale=1)
        self.model = model
        self.model.env.mode = "solve"
        
    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        # prepare
        self.from_data(points=points)
        
        # inference
        inference_start_time = time.time()
        points = to_tensor(self.points).to(self.model.env.device)
        batch_points = points.reshape(-1, batch_size, self.nodes_num, 2)

        samples = batch_points.shape[0]
        solved_tours = list()
        for idx in tqdm(range(samples), desc="Inference"):          
            _solved_tours = self.model.solve(points=batch_points[idx])
            solved_tours.append(_solved_tours)
        solved_tours = np.array(solved_tours)
        solved_tours = solved_tours.reshape(-1, self.nodes_num + 1)
        
        # select the best
        per_tours_num = solved_tours.shape[0] // points.shape[0]
        if per_tours_num > 1:
            best_tours = list()
            points = to_numpy(points)
            solved_tours = solved_tours.reshape(points.shape[0], per_tours_num, -1)
            # a problem has more than one solved tour
            for idx in range(points.shape[0]):
                evaluator = TSPEvaluator(points[idx])
                _solved_tours = solved_tours[idx]
                _solved_costs = list()
                for tour in _solved_tours:
                    _solved_costs.append(evaluator.evaluate(tour))
                _id = np.argmin(_solved_costs)
                best_tours.append(_solved_tours[_id])
            solved_tours = np.array(best_tours)
        per_tours_num = 1
        inference_end_time = time.time()
        inference_use_time = inference_end_time - inference_start_time
        print(f"Inference Time: {inference_use_time}")
        
        # local search
        if self.model.local_search is not None:
            local_search_start_time = time.time()
            solved_tours = self.model.local_search.local_search(
                solved_tours, points, None, per_tours_num
            )
            local_search_end_time = time.time()
            local_search_use_time = local_search_end_time - local_search_start_time
            print(f"Local Search Time: {local_search_use_time}")
        self.from_data(tours=solved_tours)
        return self.tours