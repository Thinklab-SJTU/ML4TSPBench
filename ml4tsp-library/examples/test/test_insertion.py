import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4tsp import *


# test file and pretrained file
NODES_NUM = 50
SPARSE_FACTOR = -1
TEST_FILE_DICT = {
    50: "test_dataset/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp500_concorde_16.546.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/tsp50_gnn.pt",
    100: "weights/tsp100_gnn.pt",
    500: "weights/tsp500_gnn.pt"
}

# main
if __name__ == "__main__":
    solver = ML4TSPNARSolver(
        model=ML4TSPGNN(
            env=ML4TSPNAREnv(nodes_num=NODES_NUM, sparse_factor=SPARSE_FACTOR, device="cuda"),
            encoder=GNNEncoder(sparse=SPARSE_FACTOR>0),
            decoder=ML4TSPNARInsertionDecoder(samples_num=1),
            local_search=None,
            pretrained_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))