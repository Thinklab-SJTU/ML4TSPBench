export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u train_nar.py \
  --wandb_logger_name "train_tsp500_diffusion" \
  --task "tsp" \
  --encoder "diffusion" \
  --num_nodes 500 \
  --sparse_factor 50 \
  --network_type "gnn" \
  --train_batch_size 8 \
  --train_file "data/uniform/train/tsp500_uniform_128k.txt" \
  --valid_file "data/uniform/valid/tsp500_uniform_val.txt" \
  --valid_samples 1280 \
  --num_workers 4 \
  --max_epochs 50 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp100_diffusion.ckpt" 