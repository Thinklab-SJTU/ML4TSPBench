export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0, 1"

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp50_dimes" \
  --task "tsp" \
  --encoder "dimes" \
  --decoding_type "greedy" \
  --time_limit 1 \
  --random_samples 100 \
  --mcts_param_t 0.1 \
  --inner_lr 0.05 \
  --active_search \
  --as_steps 100 \
  --as_samples 10000 \
  --output_channels 1 \
  --num_nodes 50 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --num_workers 4 \
  --ckpt_path "ckpt/tsp50_dimes.ckpt"