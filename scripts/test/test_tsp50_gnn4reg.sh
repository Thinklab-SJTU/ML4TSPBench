export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp50_gnn4reg" \
  --task "tsp" \
  --encoder "gnn4reg" \
  --decoding_type "greedy" \
  --local_search_type "gls" \
  --random_samples 100 \
  --mcts_param_t 0.05 \
  --time_limit 10.0 \
  --perturbation_moves 100 \
  --output_channels 1 \
  --num_nodes 50 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --regret_dir "data/uniform/regret/tsp50_uniform_regret" \
  --num_workers 4 \
  --ckpt_path "ckpt/tsp50_gnn4reg.ckpt"