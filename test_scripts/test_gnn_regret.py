import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4tsp import *


# solving settings
# set heatmap_delta=-1e5 to disable heatmap clipping for regret prediction
SOLVING_SETTINGS = "greedy_gls_1s"
SETTINGS_DICT = {
    "greedy": (ML4TSPNARGreeyDecoder(heatmap_delta=-1e5), None),
    "greedy_2opt": (ML4TSPNARGreeyDecoder(heatmap_delta=-1e5), ML4TSPNARTwoOpt()),
    "greedy_gls_1s": (ML4TSPNARGreeyDecoder(heatmap_delta=-1e5), ML4TSPNARGuidedLS(time_limit=1)),
    "greedy_gls_10s": (ML4TSPNARGreeyDecoder(heatmap_delta=-1e5), ML4TSPNARGuidedLS(time_limit=10)),
    "greedy_mcts": (ML4TSPNARGreeyDecoder(heatmap_delta=-1e5), ML4TSPNARMCTS()),
    "random10_mcts": (ML4TSPNARRandomDecoder(heatmap_delta=-1e5, samples_num=10), ML4TSPNARMCTS()),
    "random20_mcts": (ML4TSPNARRandomDecoder(heatmap_delta=-1e5, samples_num=20), ML4TSPNARMCTS()),
    "random50_mcts": (ML4TSPNARRandomDecoder(heatmap_delta=-1e5, samples_num=50), ML4TSPNARMCTS()),
    "mcts-solver-1t": (ML4TSPNARMCTSDecoder(heatmap_delta=-1e5, time_multiplier=1), None),
    "mcts-solver-5t": (ML4TSPNARMCTSDecoder(heatmap_delta=-1e5, time_multiplier=5), None),
    "mcts-solver-10t": (ML4TSPNARMCTSDecoder(heatmap_delta=-1e5, time_multiplier=10), None),
}

# test file and pretrained file
NODES_NUM = 100
SPARSE_FACTOR = -1
TEST_FILE_DICT = {
    50: "test_dataset/tsp50_concorde_5.68759.txt",
    100: "test_dataset/tsp100_concorde_7.75585.txt",
}
WEIGHT_PATH_DICT = {
    50: "weights/tsp50_gnn_reg.pt",
    100: "weights/tsp100_gnn_reg.pt",
}

# main
if __name__ == "__main__":
    print(SOLVING_SETTINGS)
    solver = ML4TSPNARSolver(
        model=ML4TSPGNNREG(
            env=ML4TSPNAREnv(nodes_num=NODES_NUM, sparse_factor=SPARSE_FACTOR, device="cuda",
                             regret_path=True),
            encoder=GNNEncoder(sparse=SPARSE_FACTOR>0, output_channels=1),
            decoder=SETTINGS_DICT[SOLVING_SETTINGS][0],
            local_search=SETTINGS_DICT[SOLVING_SETTINGS][1],
            pretrained_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))