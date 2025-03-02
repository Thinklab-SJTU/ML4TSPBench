export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp50_gnn" \
  --task "tsp" \
  --encoder "gnn" \
  --decoding_type "greedy" \
  --local_search_type "2opt" \
  --beam_size 1280 \
  --num_nodes 50 \
  --random_samples 10 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --num_workers 4 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp50_gnn.ckpt" \
  --mcts_param_t 0.001 \
  --time_limit 1 \