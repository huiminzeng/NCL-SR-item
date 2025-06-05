import numpy as np
import torch
from math import e

def utility_scores(epsilon, x, x_primes):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    num_neighbors = len(x_primes)

    cos_similarity = cos(x.unsqueeze(0), x_primes)
    util_scores = torch.exp(cos_similarity)
    sensitivity = e - 1/e
    
    util_scores = torch.exp(epsilon * util_scores / (2 * sensitivity)) / 0.01
    probs = util_scores / torch.sum(util_scores)

    return probs

def get_similar_item(x, smoothed_x, all_item_embeddings):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    x_similarity = cos(x.unsqueeze(0), all_item_embeddings)
    smoothed_x_similarity = cos(smoothed_x.unsqueeze(0), all_item_embeddings)
    original_id = torch.argmax(x_similarity, dim=-1)
    new_ids = torch.topk(smoothed_x_similarity, k=2)[1]
    if new_ids[0] == original_id:
        return new_ids[1] + 1 # convert to real item id
    else:
        return new_ids[0] + 1 # convert to real item id