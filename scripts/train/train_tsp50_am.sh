export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_ar.py \
  --wandb_logger_name "train_tsp50_am" \
  --task "tsp" \
  --encoder "am" \
  --num_nodes 50 \
  --embedding_dim 128 \
  --num_layers 3 \
  --num_heads 8 \
  --train_batch_size 640 \
  --valid_batch_size 640 \
  --train_data_size 1280 \
  --valid_file "data/uniform/valid/tsp50_uniform_val-resolve-lkh.txt" \
  --num_workers 0 \
  --max_epochs 360 \
  --monitor "val/reward" \