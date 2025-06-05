from .base import AbstractDataloader
from .split import data_split

import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils
import copy

import pdb

def get_e5_set(args, dataset):
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

    train_set = E5TrainPSDataset(args, train_data, args.bert_max_len)

    return train_set, meta

class E5TrainPSDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len):
        self.args = args
        self.max_len = max_len
        self.num_items = args.num_items
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][0][:-1] # exclude the last item, as it is the ground-truth
        return user, seq