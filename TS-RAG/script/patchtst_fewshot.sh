#!/bin/bash
# patchtst_fewshot.sh
# Few-shot fine-tuning of PatchTSTWithPrototypes on ETTh1.
# Runs all 4 prediction horizons: 96, 192, 336, 720.

export CUDA_VISIBLE_DEVICES="0"

# Data
root_path="./datasets/ETT-small/"
data_path="ETTh1.csv"
data="ett_h"
features="M"          # M: multivariate (all 7 variables)
percent=10            # 10% of training data

# Model
prototype_path="./checkpoints/prototypes_k512.npy"
seq_len=336
patch_len=16
stride=8
d_model=768           # must match prototype dimension
n_heads=8
n_layers=3
d_ff=2048
dropout=0.1

# Training
train_epochs=20
batch_size=64
learning_rate=0.0001
patience=5
tmax=10

for pred_len in 96 192 336 720; do
    model_id="ETTh1_patchtst_proto512_p${percent}_pred${pred_len}"
    echo "================================================"
    echo "Running: pred_len=${pred_len}  model_id=${model_id}"
    echo "================================================"

    python fewshot_proto.py \
        --model patchtst \
        --model_id $model_id \
        --prototype_path $prototype_path \
        --root_path $root_path \
        --data_path $data_path \
        --data $data \
        --features $features \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --percent $percent \
        --patch_len $patch_len \
        --stride $stride \
        --d_model $d_model \
        --n_heads $n_heads \
        --n_layers $n_layers \
        --d_ff $d_ff \
        --dropout $dropout \
        --train_epochs $train_epochs \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --patience $patience \
        --tmax $tmax \
        --gpu_loc 0
done

echo "All pred_len runs complete. Results in results/fewshot_evaluation/"
