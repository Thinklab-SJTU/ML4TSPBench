import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
from ml4co_kit import Trainer
from ml4tsp import ML4TSPGNNREG, ML4TSPNAREnv, GNNEncoder, ML4TSPNARGreeyDecoder


if __name__ == "__main__":
    model = ML4TSPGNNREG(
        env=ML4TSPNAREnv(
            nodes_num=50,
            mode="train",
            train_path="train_dataset/tsp50_uniform_train_128k.txt",
            val_path="train_dataset/tsp50_uniform_val-resolve-lkh.txt",
            regret_path="train_dataset/tsp50_uniform_regret",
            train_batch_size=64,
            num_workers=4,
            device="cuda",
        ),
        encoder=GNNEncoder(output_channels=1),
        decoder=ML4TSPNARGreeyDecoder(heatmap_delta=-1e5),
        learning_rate=0.0002,
    )
    trainer = Trainer(model=model, devices=[0])
    trainer.model_train()