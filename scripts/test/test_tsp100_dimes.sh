export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0, 1"

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp100_dimes" \
  --task "tsp" \
  --encoder "dimes" \
  --decoding_type "greedy" \
  --local_search_type "mcts" \
  --time_limit 1 \
  --random_samples 100 \
  --mcts_param_t 0.05 \
  --inner_lr 0.1 \
  --as_steps 100 \
  --as_samples 2000 \
  --output_channels 1 \
  --num_nodes 100 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp100_lkh_5000_7.75588.txt" \
  --num_workers 4 \
  --ckpt_path "ckpt/tsp100_dimes.ckpt"