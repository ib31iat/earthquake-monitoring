#!/bin/bash

python3 swag.py \
        --dir=".cache-bedretto/" \
        --dataset="Bedretto" \
        --batch_size=512 \
        --model="EQTransformer" \
        --dataset_path="/home/trahn/data/bedretto/" \
        --epochs=50 \
        --save_freq=5 \
        --eval_freq=10 \
        --lr_init=0.001 \
        --momentum=0.9 \
        --wd=1e-4 \
        --swa \
        --swa_start=40 \
        --swa_lr=0.02 \
        --cov_mat \
        --max_num_models=20 \
        --loss="CE" \
        --seed=42 \
        --num_workers=24 \
        --no_schedule \
        --verbose \
        --resume '.cache-bedretto/checkpoint-35.pt'

# unused option: , --resumte="T", --swa_resume="T"
