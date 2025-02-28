import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4tsp import *


# solving settings
SOLVING_SETTINGS = "greedy"
SETTINGS_DICT = {
    "greedy": ("multistart_greedy", None),
    "greedy_2opt": ("multistart_greedy", ML4TSPNARTwoOpt()),
    "sampling": ("multistart_sampling", None),
    "sampling_2opt": ("multistart_sampling", ML4TSPNARTwoOpt())
}

# test file and pretrained file
NODES_NUM = 50
TEST_FILE_DICT = {
    50: "test_dataset/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp100_concorde_7.756.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/tsp50_pomo.pt",
    100: "weights/tsp100_pomo.pt"
}

# main
if __name__ == "__main__":
    solver = ML4TSPPOMOSolver(
        model=ML4TSPPOMO(
            env=ML4TSPAREnv(nodes_num=NODES_NUM, device="cuda"),
            policy=ML4TSPPOMOPolicy(decode_type=SETTINGS_DICT[SOLVING_SETTINGS][0]),
            local_search=SETTINGS_DICT[SOLVING_SETTINGS][1],
            pretrained_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=128)
    print(solver.evaluate(calculate_gap=True))