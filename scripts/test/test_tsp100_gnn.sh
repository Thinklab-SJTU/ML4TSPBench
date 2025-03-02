export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp100_gnn" \
  --task "tsp" \
  --encoder "gnn" \
  --decoding_type "greedy" \
  --local_search_type "gls" \
  --beam_size 1280 \
  --num_nodes 100 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp100_lkh_5000_7.75588.txt" \
  --num_workers 4 \
  --ckpt_path "ckpts/tsp100_gnn.ckpt" \
  --time_limit 1 \