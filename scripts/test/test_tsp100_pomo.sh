export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_ar.py \
  --wandb_logger_name "test_tsp100_pomo" \
  --task "tsp" \
  --encoder "pomo" \
  --decoding_type "greedy" \
  --num_nodes 100 \
  --test_file "data/uniform/test/tsp100_concorde-resolve-lkh.txt" \
  --num_workers 0 \
  --ckpt_path "ckpts/tsp100_pomo.ckpt" \