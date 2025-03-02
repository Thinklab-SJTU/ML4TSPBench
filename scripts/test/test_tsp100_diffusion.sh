export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
# export WANDB_MODE=offline
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"


python -u test_nar.py \
  --wandb_logger_name "test100_diff" \
  --task "tsp" \
  --encoder "diffusion" \
  --inference_diffusion_steps 50 \
  --decoding_type "greedy" \
  --beam_size 100 \
  --random_samples 10 \
  --num_nodes 100 \
  --network_type "gnn" \
  --test_file "data/uniform/test/tsp100_lkh_5000_7.75588.txt" \
  --num_workers 4 \
  --monitor "val/loss" \
  --ckpt_path "ckpts/tsp100_diffusion.ckpt" \
  --parallel_sampling 1 \
  --mcts_smooth_v2 \
  --local_search_type "mcts" \
  --gradient_search
