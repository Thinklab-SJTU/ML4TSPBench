export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_ar.py \
  --wandb_logger_name "test_tsp50_pomo" \
  --task "tsp" \
  --encoder "pomo" \
  --decoding_type "sampling" \
  --local_search_type "2opt" \
  --num_nodes 50 \
  --test_file "data/uniform/test/tsp50_lkh_500_5.68759.txt" \
  --num_workers 0 \
  --ckpt_path "ckpts/tsp50_pomo.ckpt" 