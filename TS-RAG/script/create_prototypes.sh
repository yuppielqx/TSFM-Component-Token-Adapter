#!/bin/bash
# create_prototypes.sh
# Extract ChronosBolt encoder embeddings and cluster into prototypes.
#
# Adjust --data_source and the corresponding path flag to switch between:
#   pretrain  ->  use downloaded pretrain parquet data (recommended)
#   pkl       ->  use existing retrieval_database/*.pkl files (fallback)

export CUDA_VISIBLE_DEVICES="0"

n_clusters=512
context_length=512
max_samples=500000
batch_size=512
n_iter=100

pretrained_model_path="./checkpoints/base/"
output_path="./checkpoints/prototypes_k${n_clusters}.npy"

# ---------- choose data source ----------
data_source="parquet"      # pretrain | pkl | parquet

# pretrain mode (needs datasets/pretrain/pretrain_pairs_ctx512/)
data_path="../datasets/pretrain/pretrain_pairs_ctx${context_length}"

# pkl mode (uses existing retrieval_database/*.pkl)
pkl_dir="../retrieval_database/"
pkl_stride=64

# parquet mode (uses single retrieval_database parquet file)
parquet_path="../datasets/pretrain/retrieval_database_512.parquet"
# ----------------------------------------

if [ "$data_source" == "pretrain" ]; then
    python create_prototypes.py \
        --data_source pretrain \
        --data_path $data_path \
        --pretrained_model_path $pretrained_model_path \
        --context_length $context_length \
        --max_samples $max_samples \
        --n_clusters $n_clusters \
        --n_iter $n_iter \
        --kmeans_backend faiss \
        --output_path $output_path \
        --batch_size $batch_size \
        --gpu_loc 0
elif [ "$data_source" == "pkl" ]; then
    python create_prototypes.py \
        --data_source pkl \
        --pkl_dir $pkl_dir \
        --pkl_stride $pkl_stride \
        --pretrained_model_path $pretrained_model_path \
        --context_length $context_length \
        --max_samples $max_samples \
        --n_clusters $n_clusters \
        --n_iter $n_iter \
        --kmeans_backend faiss \
        --output_path $output_path \
        --batch_size $batch_size \
        --gpu_loc 0
else
    python create_prototypes.py \
        --data_source parquet \
        --parquet_path $parquet_path \
        --pretrained_model_path $pretrained_model_path \
        --context_length $context_length \
        --max_samples $max_samples \
        --n_clusters $n_clusters \
        --n_iter $n_iter \
        --kmeans_backend sklearn \
        --output_path $output_path \
        --batch_size $batch_size \
        --gpu_loc 0
fi
