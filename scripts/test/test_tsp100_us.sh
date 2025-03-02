export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp100_us" \
  --task "tsp" \
  --encoder "us" \
  --network_type "sag" \
  --decoding_type "rmcts" \
  --random_samples 10 \
  --num_nodes 100 \
  --mcts_param_t 0.01 \
  --input_dim 2 \
  --hidden_dim 64 \
  --output_channels 100 \
  --num_layers 3 \
  --num_heads 8 \
  --test_file "data/uniform/test/tsp100_lkh_5000_7.75588.txt" \
  --num_workers 4 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp100_us.ckpt" \
