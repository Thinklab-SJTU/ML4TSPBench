export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0, 1"

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_nar.py \
  --wandb_logger_name "train_tsp50_gnn4reg" \
  --task "tsp" \
  --encoder "gnn4reg" \
  --num_nodes 50 \
  --network_type "gnn" \
  --input_dim 2 \
  --embedding_dim 128 \
  --hidden_dim 256 \
  --output_channels 1 \
  --num_layers 12 \
  --num_heads 8 \
  --train_batch_size 64 \
  --train_file "data/uniform/train/tsp50_uniform_train_1.28M.txt" \
  --valid_file "data/uniform/valid/tsp50_uniform_val-resolve-lkh.txt" \
  --regret_dir "data/uniform/regret/tsp50_uniform_regret" \
  --valid_samples 1280 \
  --num_workers 4 \
  --max_epochs 100 \
  --monitor "val/gap" \