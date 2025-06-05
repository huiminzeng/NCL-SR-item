from .base import AbstractDataloader
from .split import data_split

from .contrastive_utils import *

import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils
import copy

import pdb

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2', force_download=False)
tokenizer.truncation_side = 'left'

def collate_fn_non_cl(data):
    batch_size = len(data)
    user_id, seq, purchase_text, input_text, target_text = zip(*data)

    purchase_text = sum(purchase_text, [])
    purchase_tokens = tokenizer(purchase_text, max_length=256, truncation=True, padding=True, return_tensors="pt")

    return user_id, seq, (purchase_tokens['input_ids'], purchase_tokens['token_type_ids'], purchase_tokens['attention_mask']), \
            input_text, target_text, batch_size

def collate_fn_val(data):
    batch_size = len(data)
    purchase_text = sum(data, [])
    purchase_tokens = tokenizer(purchase_text, max_length=256, truncation=True, padding=True, return_tensors="pt")

    return purchase_tokens['input_ids'], purchase_tokens['token_type_ids'], purchase_tokens['attention_mask'], batch_size

def collate_fn_test(data):
    return data

def get_e5_data_non_cl(args, dataset, tokenizer):
    dataset = dataset.load_dataset()
    user2items = dataset['user2items']
    user2scores = dataset['user2scores']
    # user2sessionIDs = dataset['user2sessionIDs']
    # user2sessions = dataset['user2sessions']
    umap = dataset['umap']
    smap = dataset['smap']
    meta = dataset['meta']
    user_count = len(umap)
    item_count = len(smap)
    
    train_data, val_data, test_data = data_split(user2items, user2scores, user_count)
    args.num_users = user_count
    args.num_items = item_count

    train_set = E5NonCLTrainDataset(args, train_data, args.bert_max_len, meta, tokenizer)
    val_set = E5ValidDataset(args, val_data, args.bert_max_len, meta, tokenizer)
    test_set = E5TestDataset(args, test_data, args.bert_max_len)

    return train_set, val_set, test_set, meta

class E5NonCLTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, meta, tokenizer):
        self.args = args
        self.max_len = max_len
        self.num_items = args.num_items

        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())

        self.meta = meta
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][0]

        input_text = get_input_seq_template(self.args, seq[:-1], self.meta)
        target_text = get_target_item_template(self.args, seq[-1], self.meta)

        anchor_purchases_text, positive_purchases_text, negative_purchases_text = sampling_contrastive_purchases(seq, 'train', index, self.meta, self.args)
        purchase_text = [anchor_purchases_text, positive_purchases_text]
        purchase_text += negative_purchases_text

        return user, seq, purchase_text, input_text, target_text
    

class E5ValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, meta, tokenizer):
        self.args = args
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.meta = meta
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][0]

        # sampling contrastive purchases
        anchor_purchases_text, positive_purchases_text, negative_purchases_text = sampling_contrastive_purchases(seq, 'val', index, self.meta, self.args)
        purchase_text = [anchor_purchases_text, positive_purchases_text]
        purchase_text += negative_purchases_text
        # purchases_tokens = tokenize_512(purchase_text, self.tokenizer)

        return purchase_text
    
class E5TestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len):
        self.args = args
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][0]

        return seq