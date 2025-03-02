export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_nar.py \
  --wandb_logger_name "train_tsp100_us_sag" \
  --task "tsp" \
  --encoder "us" \
  --num_nodes 100 \
  --network_type "sag" \
  --input_dim 2 \
  --hidden_dim 64 \
  --output_channels 100 \
  --num_layers 3 \
  --num_heads 8 \
  --train_batch_size 16 \
  --train_file "data/uniform/train/tsp100_uniform_2k.txt" \
  --valid_file "data/uniform/valid/tsp100_uniform_val_lkh5k.txt" \
  --learning_rate 0.003 \
  --valid_samples 1280 \
  --num_workers 4 \
  --max_epochs 100 \
  --monitor "val/loss" \