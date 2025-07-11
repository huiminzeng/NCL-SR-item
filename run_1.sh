TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/train_stage_1.py --mce_gamma_a 0.7 --mce_gamma_u 1.3 --mce_gamma_s 0.1 --num_replace 3 --dataset_code games --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --print_freq 200 --epochs 5 \
# & TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=3  python src/train_stage_1.py --mce_gamma_a 0.7 --mce_gamma_u 1.3 --mce_gamma_s 0.1 --num_replace 3 --dataset_code beauty --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --print_freq 200 --epochs 5;    

# TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/train_stage_2.py --mce_gamma_a 0.7 --mce_gamma_u 1.3 --mce_gamma_s 0.1 --num_replace 3 --dataset_code beauty --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --print_freq 45 --epochs 3;
# TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=2  python src/eval_e5.py       --mce_gamma_a 0.7 --mce_gamma_u 1.3 --mce_gamma_s 0.1 --num_replace 3 --dataset_code beauty --test_batch_size 32;

# TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/train_stage_1.py --mce_gamma_a 0.9 --mce_gamma_u 1.3 --num_replace 3 --dataset_code auto --lr 5e-5 --train_batch_size 4 --val_batch_size 4 --print_freq 12 --epochs 5;
# TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/train_stage_2.py --mce_gamma_a 0.9 --mce_gamma_u 1.3 --num_replace 3 --dataset_code auto --lr 5e-6 --train_batch_size 4 --val_batch_size 4 --print_freq 12 --epochs 3;
# TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=0  python src/eval_e5.py       --mce_gamma_a 0.9 --mce_gamma_u 1.3 --num_replace 3 --dataset_code auto --test_batch_size 32;
