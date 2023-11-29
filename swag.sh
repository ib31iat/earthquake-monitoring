#!/bin/bash

python3 swag.py \
        --dir=".cache/" \
        --dataset="STEAD" \
        --batch_size=512 \
        --model="EQTransformer" \
        --dataset_path="/data/STEAD_dataset/.seisbench/datasets/stead/" \
        --epochs=20 \
        --save_freq=5 \
        --eval_freq=10 \
        --lr_init=0.001 \
        --momentum=0.9 \
        --wd=1e-4 \
        --swa \
        --swa_start=15 \
        --swa_lr=0.02 \
        --cov_mat \
        --max_num_models=20 \
        --loss="CE" \
        --seed=42 \
        --num_workers=24 \
        --no_schedule \
        --verbose

# unused option: , --resumte="T", --swa_resume="T"
