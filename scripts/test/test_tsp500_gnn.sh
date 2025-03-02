export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"


python -u test_nar.py \
    --wandb_logger_name "test_tsp500_gnn" \
    --task "tsp" \
    --encoder "gnn" \
    --decoding_type "greedy" \
    --beam_probs_type "logits" \
    --beam_size 10 \
    --sparse_factor 50 \
    --num_nodes 500 \
    --random_samples 100 \
    --network_type "gnn" \
    --test_file "data/uniform/test/tsp500_lkh_50000_16.54811.txt" \
    --num_workers 4 \
    --monitor "val/loss" \
    --ckpt_path "ckpts/tsp500_gnn.ckpt" \
    --time_limit 1