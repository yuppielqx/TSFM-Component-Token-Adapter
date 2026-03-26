#!/bin/bash
# train_prototypes.sh
# Learn prototype matrix end-to-end via frozen ChronosBolt backbone.
# Output: checkpoints/prototypes_learned.npy  (shape: n_clusters × 768)
# Use with: fewshot_proto.py --prototype_path ./checkpoints/prototypes_learned.npy

export CUDA_VISIBLE_DEVICES="2"

python train_prototypes.py \
    --pretrained_model_path ./checkpoints/base/ \
    --parquet_path          ./datasets/pretrain/retrieval_database_512.parquet \
    --n_clusters            512 \
    --n_heads               8 \
    --proto_dropout         0.1 \
    --train_epochs          5 \
    --batch_size            128 \
    --num_workers           4 \
    --learning_rate         1e-4 \
    --weight_decay          0.01 \
    --grad_clip             1.0 \
    --log_every             200 \
    --output_path           ./checkpoints/prototypes_learned.npy \
    --gpu_loc               0

echo "Prototype training complete."
echo "Saved to: ./checkpoints/prototypes_learned.npy"

# ---------------------------------------------------------------------------
# Recovery task: RQ-VAE prototype training with masked reconstruction
# ---------------------------------------------------------------------------
# Uncomment the block below (and comment the block above) to train with
# the RQ-VAE recovery objective instead of the forecasting objective.
#
# python train_prototypes.py \
#     --task              recovery \
#     --pretrained_model_path ./checkpoints/base/ \
#     --parquet_path          ./datasets/pretrain/retrieval_database_512.parquet \
#     --n_proto               256 \
#     --n_rq_stages           4 \
#     --beta                  0.25 \
#     --mask_ratio            0.1 \
#     --train_epochs          20 \
#     --patience              3 \
#     --batch_size            128 \
#     --num_workers           4 \
#     --learning_rate         1e-4 \
#     --weight_decay          0.01 \
#     --grad_clip             1.0 \
#     --log_every             200 \
#     --output_path           ./checkpoints/prototypes_rqvae.pt \
#     --checkpoint_dir        ./checkpoints/proto_ckpt/ \
#     --gpu_loc               0
#
# echo "RQ-VAE prototype training complete."
# echo "Saved to: ./checkpoints/prototypes_rqvae.pt"
