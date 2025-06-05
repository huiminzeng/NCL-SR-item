import numpy as np
import random
import torch
import argparse

RAW_DATASET_ROOT_FOLDER = '../../Data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

def set_template(args):
    if args.dataset_code == 'ml-100k':
        args.bert_max_len = 200
    else:
        args.bert_max_len = 50
    
    if args.dataset_code == 'beauty':
        args.session_interval = 30000000
    
    # if args.dataset_code == 'ml-100k':
    #     batch = 16 
    # else:
    #     batch = 4

    # args.train_batch_size = args.batch
    # args.val_batch_size = args.batch
    # args.test_batch_size = args.batch

    args.optimizer = 'AdamW'
    args.weight_decay = 0.01
    args.enable_lr_schedule = True
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--session_interval', type=int, default=None)
################
# Dataloader
################
parser.add_argument('--batch', type=int, default=None)
parser.add_argument('--train_batch_size', type=int, default=None)
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--test_batch_size', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=16)

parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=False)
parser.add_argument('--warmup_steps', type=int, default=100)
parser.add_argument('--margin', type=float, default=0.1)
parser.add_argument('--print_freq', type=int, default=1000)

################
# Evaluation
################
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--export_dir', type=str, default=None)

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.2)
parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# Inference Ensemble
################
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--ensembled_tokp', type=int, default=20)
parser.add_argument('--lambda_ensemble', type=float, default=0.2)
parser.add_argument('--topk', type=int, default=20)

################
# Differemtial Privacy
################
parser.add_argument('--min_cos_sim', type=float, default=0.95)
parser.add_argument('--num_replace', type=int, default=1)
parser.add_argument('--dp_epsilon', type=float, default=0.8)
parser.add_argument('--topk_user', type=int, default=5)
parser.add_argument('--uncertainty', type=float, default=0.7)

################
# DeepSpeed
################
parser.add_argument('--ds_mode', type=str, default='1_gpu')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
# parser.add_argument('--dtype', type=str, default='bf16')
parser.add_argument('--dtype', type=str, default=None)
parser.add_argument('--stage', type=int, default=2)
parser.add_argument('--local_device', default=None)

################
# Contrastive Pretraining
################
parser.add_argument('--lambda_loss_1', type=float, default=1)    # contrastive loss for purchase
parser.add_argument('--lambda_loss_2', type=float, default=0)    # contrastive loss for behaviors
parser.add_argument('--lambda_loss_3', type=float, default=0)    # self contrastive for augmented session
parser.add_argument('--num_non_purchase', type=int, default=41)     # non-purchased items
parser.add_argument('--num_pos_behavior', type=float, default=0.5)  # positive behavior sampling (fraction of the seq lengths)
parser.add_argument('--num_neg_behavior', type=int, default=3)      # negative behavior sampling (per positive behavior)

################
# Matrix-SSL
################
parser.add_argument('--mce_lamda', type=float, default=0.5)
parser.add_argument('--mce_gamma_a', type=float, default=None)
parser.add_argument('--mce_gamma_u', type=float, default=None)
parser.add_argument('--mce_gamma_s', type=float, default=None)
parser.add_argument('--mce_mu', type=float, default=0.995)
parser.add_argument('--mce_order', type=int, default=4)
parser.add_argument('--teacher_momentum', default=0.996, type=float, metavar='M',
                    help='momentum of teacher update')
parser.add_argument('--lambda_loss_mssl', type=float, default=0.1)  

args = parser.parse_args()
