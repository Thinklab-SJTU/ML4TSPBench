import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4tsp import *


# solving settings
SOLVING_SETTINGS = "greedy"
SETTINGS_DICT = {
    "greedy": (ML4TSPNARGreeyDecoder(), None),
    "greedy_2opt": (ML4TSPNARGreeyDecoder(), ML4TSPNARTwoOpt()),
    "greedy_mcts": (ML4TSPNARGreeyDecoder(), ML4TSPNARMCTS()),
}

# test file and pretrained file
NODES_NUM = 50
SPARSE_FACTOR = -1
TEST_FILE_DICT = {
    50: "test_dataset/tsp50_concorde_5.68883.txt",
    100: "test_dataset/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp500_concorde_16.58356.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/tsp50_diffusion.pt",
    100: "weights/tsp100_diffusion.pt",
    500: "weights/tsp500_diffusion.pt"
}

GRADIENT_SEARCH_FLAG = True

# main
if __name__ == "__main__":
    if GRADIENT_SEARCH_FLAG:
        solver_type = ML4TSPT2TSolver
    else:
        solver_type = ML4TSPNARSolver
    
    solver = solver_type(
        model=ML4TSPDiffusion(
            env=ML4TSPNAREnv(nodes_num=NODES_NUM, sparse_factor=SPARSE_FACTOR, device="cuda"),
            encoder=GNNEncoder(sparse=SPARSE_FACTOR>0, time_embed_flag=True),
            decoder=SETTINGS_DICT[SOLVING_SETTINGS][0],
            local_search=SETTINGS_DICT[SOLVING_SETTINGS][1],
            pretrained_path=WEIGHT_PATH_DICT[NODES_NUM],
            inference_diffusion_steps=50
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))