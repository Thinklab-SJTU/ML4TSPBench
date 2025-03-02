export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_nar.py \
  --wandb_logger_name "train_tsp500_gnn_wise" \
  --task "tsp" \
  --encoder "gnn-wise" \
  --num_nodes 500 \
  --sparse_factor 50 \
  --network_type "gnn" \
  --input_dim 2 \
  --embedding_dim 128 \
  --hidden_dim 256 \
  --output_channels 2 \
  --num_layers 12 \
  --num_heads 8 \
  --train_batch_size 8 \
  --train_file "data/uniform/train/tsp500_uniform_128k.txt" \
  --valid_file "data/uniform/valid/tsp500_uniform_val.txt" \
  --valid_samples 1280 \
  --num_workers 4 \
  --max_epochs 50 \
  --monitor "val/loss" \
  --ckpt_path "train_ckpts/train_tsp500_gnn_wise/71lcdx6p/epoch=22-step=184000.ckpt" \