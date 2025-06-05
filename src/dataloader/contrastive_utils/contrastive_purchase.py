import numpy as np
from trainer import *
from .template import *

def sampling_contrastive_purchases(seq, mode, seed, meta, args):
    """
    this function constructs contrastive pairs based on the criterion:
    entire history and ground-truth item as positive pair
    entire history and non-ground-truth item as negative pair
    """
    
    items_pool = list(range(1, args.num_items))
    history_text = get_input_seq_template(args, seq[:-1], meta)
    target_text = get_target_item_template(args, seq[-1], meta)

    seen_items = set(seq)
    negative_pool = set(items_pool) - seen_items
    negative_pool = list(negative_pool)
    if mode == 'train':
        np.random.seed(seed)
        negative_sample = np.random.choice(negative_pool, args.num_non_purchase, replace=False)

    elif mode == 'val':
        np.random.seed(seed)
        negative_sample = np.random.choice(negative_pool, args.num_non_purchase, replace=False)

    temp_texts = []
    for neg_sam in negative_sample:
        negative_text = get_target_item_template(args, neg_sam, meta)
        temp_texts.append(negative_text)

    return history_text, target_text, temp_texts
