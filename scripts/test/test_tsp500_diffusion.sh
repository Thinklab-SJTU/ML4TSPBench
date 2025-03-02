export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_MODE=offline
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u test_nar.py \
  --wandb_logger_name "test_tsp500_diffusion" \
  --task "tsp" \
  --encoder "diffusion" \
  --inference_diffusion_steps 50 \
  --decoding_type "greedy" \
  --local_search_type "2opt" \
  --beam_size 1280 \
  --sparse_factor 50 \
  --num_nodes 500 \
  --random_samples 10 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp500_lkh_50000_16.54811.txt" \
  --num_workers 4 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp500_diffusion.ckpt" \
  --parallel_sampling 1 \
  --steps_inf 10 \
  --mcts_smooth_v2 \
  --gradient_search