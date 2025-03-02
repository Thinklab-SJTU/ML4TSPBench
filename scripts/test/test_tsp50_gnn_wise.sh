export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp50_gnn_wise" \
  --task "tsp" \
  --encoder "gnn-wise" \
  --decoding_type "greedy" \
  --local_search_type "mcts" \
  --beam_size 1280 \
  --num_nodes 50 \
  --mcts_param_t 0.001 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --num_workers 4 \
  --ckpt_path "ckpts/tsp50_gnn_wise.ckpt" \
