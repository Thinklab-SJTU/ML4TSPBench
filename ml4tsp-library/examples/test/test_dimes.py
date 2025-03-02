import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4tsp import *


# solving settings
SOLVING_SETTINGS = "greedy_mcts"
SETTINGS_DICT = {
    "greedy": (ML4TSPNARGreeyDecoder(), None),
    "greedy_2opt": (ML4TSPNARGreeyDecoder(), ML4TSPNARTwoOpt()),
    "greedy_mcts": (ML4TSPNARGreeyDecoder(), ML4TSPNARMCTS()),
    "beam-1280": (ML4TSPNARBeamDecoder(beam_size=1280, return_best=True), None),
    "beam-1280_2opt": (ML4TSPNARBeamDecoder(beam_size=1280, return_best=True), None),
    "beam10_mcts": (ML4TSPNARBeamDecoder(beam_size=10, return_best=False), ML4TSPNARMCTS()),
    "beam20_mcts": (ML4TSPNARBeamDecoder(beam_size=20, return_best=False), ML4TSPNARMCTS()),
    "beam50_mcts": (ML4TSPNARBeamDecoder(beam_size=50, return_best=False), ML4TSPNARMCTS()),
    "sampling10_mcts": (ML4TSPNARSamplingDecoder(samples_num=10), ML4TSPNARMCTS()),
    "sampling20_mcts": (ML4TSPNARSamplingDecoder(samples_num=20), ML4TSPNARMCTS()),
    "sampling50_mcts": (ML4TSPNARSamplingDecoder(samples_num=50), ML4TSPNARMCTS()),
    "random10_mcts": (ML4TSPNARRandomDecoder(samples_num=10), ML4TSPNARMCTS()),
    "random20_mcts": (ML4TSPNARRandomDecoder(samples_num=20), ML4TSPNARMCTS()),
    "random50_mcts": (ML4TSPNARRandomDecoder(samples_num=50), ML4TSPNARMCTS()),
    "mcts-solver-1t": (ML4TSPNARMCTSDecoder(time_multiplier=1), None),
    "mcts-solver-5t": (ML4TSPNARMCTSDecoder(time_multiplier=5), None),
    "mcts-solver-10t": (ML4TSPNARMCTSDecoder(time_multiplier=10), None)
}

# test file and pretrained file
NODES_NUM = 50
TEST_FILE_DICT = {
    50: "test_dataset/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp500_concorde_16.546.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/tsp50_dimes.pt",
    100: "weights/tsp100_dimes.pt",
    500: "weights/tsp500_dimes.pt"
}

# main
if __name__ == "__main__":
    solver = ML4TSPNARSolver(
        model=ML4TSPDIMES(
            env=ML4TSPNAREnv(nodes_num=NODES_NUM, sparse_factor=-1, device="cuda"),
            encoder=GNNEncoder(output_channels=1, sparse=False),
            decoder=SETTINGS_DICT[SOLVING_SETTINGS][0],
            local_search=SETTINGS_DICT[SOLVING_SETTINGS][1],
            pretrained_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))