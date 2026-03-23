#!/bin/bash
# fewshot.sh
# Few-shot fine-tuning of ChronosBoltWithPrototypes on downstream datasets.

export CUDA_VISIBLE_DEVICES="0"

prototype_path="./checkpoints/prototypes_k512.pt"
pretrained_model_path="./checkpoints/base/"
seq_len=512
pred_len=96
batch_size=64
learning_rate=1e-4
train_epochs=10
percent=10        # % of training set to use

datasets="ETTh1 ETTh2"

for dataset in $datasets; do
    if [ "$dataset" == "ETTh1" ] || [ "$dataset" == "ETTh2" ]; then
        data="ett_h"
        root_path="../datasets/ETT-small/"
        freq="h"
    elif [ "$dataset" == "ETTm1" ] || [ "$dataset" == "ETTm2" ]; then
        data="ett_m"
        root_path="../datasets/ETT-small/"
        freq="t"
    else
        data="custom"
        root_path="../datasets/${dataset}/"
        freq="h"
    fi

    model_id="${dataset}_fewshot_proto512_p${percent}_pred${pred_len}"

    python fewshot.py \
        --model_id $model_id \
        --pretrained_model_path $pretrained_model_path \
        --prototype_path $prototype_path \
        --root_path $root_path \
        --data_path "${dataset}.csv" \
        --data $data \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --percent $percent \
        --train_epochs $train_epochs \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --freq $freq \
        --patience 3 \
        --tmax 10 \
        --gpu_loc 0 \
        --save_file_name "fewshot_results.txt"

    echo "Done: $dataset"
done
