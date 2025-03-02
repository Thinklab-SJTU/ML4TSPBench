export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="3"

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_nar.py \
  --wandb_logger_name "train_tsp50_dimes" \
  --task "tsp" \
  --encoder "dimes" \
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
  --valid_samples 1280 \
  --num_workers 4 \
  --max_steps 200 \
  --strategy "auto" \
  --every_n_train_steps 10 \
  --val_check_interval 5 \
  --log_every_n_steps 1 \
  --inner_epochs 100 \
  --inner_samples 2000 \
  --learning_rate 1e-3 \
  --inner_lr 1e-1 \
  --monitor "val/gap" \