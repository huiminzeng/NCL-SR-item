#!/bin/bash

mce_gamma_a_list=(0.1 0.3 0.5 0.7 0.9)
mce_gamma_u_list=(0.1 0.3 0.5 0.7 0.9)
datasets=(beauty games sports toys_new office auto)

for mce_gamma_a in "${mce_gamma_a_list[@]}"; do
  for mce_gamma_u in "${mce_gamma_u_list[@]}"; do
      for dataset in "${datasets[@]}"; do
        TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.0 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=1  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.05 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.1 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=3  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.3 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=4  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.5 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=5  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.7 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=6  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.9 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=7  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.0 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=8  python src/train_stage_1.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.1 --num_replace 3 --dataset_code $dataset --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --epochs 5; 
    done
  done
done

for mce_gamma_a in "${mce_gamma_a_list[@]}"; do
  for mce_gamma_u in "${mce_gamma_u_list[@]}"; do
      for dataset in "${datasets[@]}"; do
        TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.0 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=1  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.05 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.1 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=3  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.3 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=4  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.5 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=5  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.7 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=6  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.9 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=7  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.0 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=8  python src/train_stage_2.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.1 --num_replace 3 --dataset_code $dataset --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --epochs 3; 
    done
  done
done

for mce_gamma_a in "${mce_gamma_a_list[@]}"; do
  for mce_gamma_u in "${mce_gamma_u_list[@]}"; do
      for dataset in "${datasets[@]}"; do
        TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.0 --num_replace 3 --dataset_code $dataset --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=1  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.05 --num_replace 3 --dataset_code $dataset --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.1 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=3  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.3 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=4  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.5 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=5  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.7 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=6  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 0.9 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=7  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.0 --num_replace 3 --dataset_code $dataset  --test_batch_size 32 \
        & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=8  python src/eval_e5.py --mce_gamma_a $mce_gamma_a --mce_gamma_u $mce_gamma_u --mce_gamma_s 1.1 --num_replace 3 --dataset_code $dataset  --test_batch_size 32;
      done
  done
done
