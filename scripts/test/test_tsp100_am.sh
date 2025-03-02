export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_ar.py \
  --wandb_logger_name "test_tsp100_am" \
  --task "tsp" \
  --encoder "am" \
  --decoding_type "multistart_sampling" \
  --local_search_type "2opt" \
  --num_nodes 100 \
  --test_file "data/uniform/test/tsp100_lkh_5000_7.75588.txt" \
  --num_workers 0 \
  --ckpt_path "ckpts/tsp100_am.ckpt"