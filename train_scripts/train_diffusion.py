import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
from ml4co_kit import Trainer
from ml4tsp import ML4TSPDiffusion, ML4TSPNAREnv, GNNEncoder, ML4TSPNARGreeyDecoder


if __name__ == "__main__":
    model = ML4TSPDiffusion(
        env=ML4TSPNAREnv(
            nodes_num=50,
            mode="train",
            train_path="data/tsp50_lkh_500_5.68759.txt",
            val_path="data/tsp50_lkh_500_5.68759.txt",
            train_batch_size=64,
            num_workers=4,
            device="cuda",
        ),
        encoder=GNNEncoder(sparse=False, time_embed_flag=True),
        decoder=ML4TSPNARGreeyDecoder(heatmap_delta=-1e5),
        learning_rate=0.0002,
    )
    
    # model = ML4TSPDiffusion(
    #     env=ML4TSPNAREnv(
    #         nodes_num=50,
    #         mode="train",
    #         # train_path="data/tsp50_lkh_500_5.68759.txt",
    #         train_path="data/tsp500_lkh_10000_16.55283.txt",
    #         # val_path="data/tsp50_lkh_500_5.68759.txt",
    #         val_path="data/tsp500_lkh_10000_16.55283.txt",
    #         sparse_factor=50,
    #         train_batch_size=16,
    #         num_workers=4,
    #         device="cuda",
    #     ),
    #     encoder=GNNEncoder(sparse=True),
    #     decoder=ML4TSPNARGreeyDecoder(heatmap_delta=-1e5),
    #     learning_rate=0.0002,
    # )
    trainer = Trainer(model=model, devices=[0])
    trainer.model_train()