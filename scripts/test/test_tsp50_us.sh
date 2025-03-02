export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp50_gnn-wise" \
  --task "tsp" \
  --encoder "us" \
  --network_type "sag" \
  --decoding_type "greedy" \
  --local_search_type "2opt" \
  --random_samples 10 \
  --num_nodes 50 \
  --input_dim 2 \
  --hidden_dim 64 \
  --output_channels 50 \
  --num_layers 3 \
  --num_heads 8 \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --num_workers 4 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp50_us.ckpt" \
  --mcts_smooth \
  --mcts_param_t 0.01 \
