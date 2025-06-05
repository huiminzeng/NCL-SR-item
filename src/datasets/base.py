import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'


    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    def all_raw_file_names(cls):
        return []

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 1 or self.min_uc > 1:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
                if self.min_sc > 1:
                    item_sizes = df.groupby('sid').size()
                    good_items = item_sizes.index[item_sizes >= self.min_sc]
                    df = df[df['sid'].isin(good_items)]

                if self.min_uc > 1:
                    user_sizes = df.groupby('uid').size()
                    good_users = user_sizes.index[user_sizes >= self.min_uc]
                    df = df[df['uid'].isin(good_users)]

                item_sizes = df.groupby('sid').size()
                good_items = item_sizes.index[item_sizes >= self.min_sc]
                user_sizes = df.groupby('uid').size()
                good_users = user_sizes.index[user_sizes >= self.min_uc]
        return df
    
    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(list(np.unique(df['uid'])), start=1)}
        smap = {s: i for i, s in enumerate(list(np.unique(df['sid'])), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        print('Splitting')
        df = df.sort_values(['uid', 'timestamp'])
        user_group = df.groupby('uid')
        df_prev = df.shift()

        # df[df['uid']==10]
        is_new_session = (df.uid != df_prev.uid) | (df.timestamp - df_prev.timestamp > self.args.session_interval)
        session_id = is_new_session.cumsum() - 1
        df['session_id'] = session_id
        user2items = user_group.progress_apply(lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
        user2scores = user_group.progress_apply(lambda d: list(d.sort_values(by=['timestamp', 'sid'])['rating']))
        user2sessionID_temp = user_group.progress_apply(lambda d: list(d.sort_values(by=['timestamp', 'sid'])['session_id']))
        user2sessionID = {}
        user2sessions = {}
        for i in range(user_count):
            user = i + 1
            sessionIDs = user2sessionID_temp[user]
            seq = user2items[user]
            sessionIDs = np.array(sessionIDs) - np.min(sessionIDs)
            sessionIDs = list(sessionIDs)
            sessionIDs_arr = np.array(sessionIDs)

            user2sessionID[user] = sessionIDs
            user2sessions[user] = []
            for j in sorted(set(sessionIDs)):
                seq_arr = np.array(seq)
                session = seq_arr[np.where(sessionIDs_arr == j)[0]].tolist()
                user2sessions[user].append(session)

        return user2items, user2scores, user2sessionID, user2sessions

    def group_session(self, user2items):
        pass

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')
