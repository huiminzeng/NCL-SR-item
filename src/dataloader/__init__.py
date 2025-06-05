from datasets import dataset_factory

from .e5_non_cl import *
from .e5_preprocessing import get_e5_set
from .contrastive_utils import *

def dataloader_factory_non_cl(args, tokenizer):
    dataset = dataset_factory(args)
    train_set, val_set, test_set, meta = get_e5_data_non_cl(args, dataset, tokenizer)
    return train_set, val_set, test_set, meta


def get_datasets(args):
    dataset = dataset_factory(args)
    train_set, meta = get_e5_set(args, dataset)
    return train_set, meta