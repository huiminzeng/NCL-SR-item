import torch
import tqdm
import copy
import numpy as np
import torch.nn.functional as F

from .dp import *

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_input_item_template(args, item, meta):
    try:
        title = meta[item][0]
    except:
        title = meta[item[0]][0]
    category = meta[item][2]

    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new', 'sports', 'office']:
        prompt = "query: "
        prompt += "'"
        prompt += title
        prompt += "', a product from "
        prompt += category
        prompt += ' category.'

    return prompt

def calculate_all_item_embeddings(model, meta, args):
    candidate_prompts = []
    for item in range(1, args.num_items+1):
        candidate_text = get_input_item_template(args, item, meta)
        candidate_prompts.append(candidate_text)
    candidate_embeddings =[]
    
    with torch.no_grad():
        for i in range(0, args.num_items, args.val_batch_size):
            input_prompts = candidate_prompts[i: i + args.val_batch_size]
        
            input_tokens = model.tokenizer(input_prompts, max_length=64, truncation=True, padding=True, return_tensors="pt")
            input_tokens = {k: v.to(args.device) for k, v in input_tokens.items()}

            outputs = model.model(**input_tokens)
            embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'].to(args.device))
            embeddings = F.normalize(embeddings, dim=-1)

            candidate_embeddings.append(embeddings)

        candidate_embeddings = torch.cat(candidate_embeddings)
        
    return candidate_embeddings


def build_1_order_synonym_dict(min_cos_sim, item_embeddings, min_items=2, max_items=25):
    synonym_matrix = []
    synonym_dict_1_order = {}
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(len(item_embeddings)):
        # item id starts from 1, 
        # but item_embeddings starts with feature id 0 for item 1
        item_id = i + 1
        item_vec = item_embeddings[i]
        scores = cos(item_vec, item_embeddings)

        # indices = (scores >= min_cos_sim).nonzero(as_tuple=False)
        # if len(indices) <= min_items:

        _, indices = torch.sort(scores, descending=True)
        indices = indices[:(min_items+1)]

        # save the real item id 
        # not the feature id
        synonym_items = (indices+1).squeeze().cpu().numpy().tolist() 

        if item_id in synonym_items:
            synonym_items.remove(item_id)

        if len(synonym_items) > max_items:
            synonym_items = np.random.choice(synonym_items, max_items, replace=False).tolist()
            
        synonym_dict_1_order[item_id] = synonym_items
        synonym_matrix.append(synonym_items[:2])
    
    synonym_matrix.append([len(item_embeddings)+1]*2)

    return synonym_matrix, synonym_dict_1_order
    
def build_2_order_synonym_dict(synonym_matrix, synonym_dict_1_order, max_items=25):
    synonym_dict_2_order = {}
    counter = 0
    for item_id in synonym_dict_1_order.keys():
        synonym_items_2_order = []
        synonyms = synonym_dict_1_order[item_id]
        for synonym in synonyms:
            synonym_items_2_order += synonym_dict_1_order[synonym]
        synonym_items_2_order = set(synonym_items_2_order)
        if item_id in synonym_items_2_order:
            synonym_items_2_order.remove(item_id)
        
        for synonym in synonyms:
            if synonym in synonym_items_2_order:
                synonym_items_2_order.remove(synonym)
        if len(synonym_items_2_order) > max_items:
            synonym_items_2_order = list(synonym_items_2_order)
            synonym_items_2_order = np.random.choice(
                synonym_items_2_order, max_items, replace=False).tolist()
        synonym_dict_2_order[item_id] = list(synonym_items_2_order)

        if len(list(synonym_items_2_order)) == 1 or len(list(synonym_items_2_order)) == 2:
            temp = list(synonym_items_2_order) + list(np.random.choice(list(synonym_items_2_order), 3 - len(list(synonym_items_2_order))))
            synonym_matrix[counter] += temp
        elif len(list(synonym_items_2_order)) == 0:
            temp = list(np.random.choice(range(1, len(synonym_matrix)+1), 3))
            synonym_matrix[counter] += temp
        else:
            synonym_matrix[counter] += list(synonym_items_2_order)[:3]
    
        counter += 1

    synonym_matrix[counter] += [len(synonym_matrix)] * 3
    synonym_matrix = torch.cat([torch.zeros(1,5), torch.Tensor(synonym_matrix)], dim=0).cuda().long()

    return synonym_matrix

def calculate_substitutions(batch, synonym_matrix, all_item_embeddings_e5, args):
    batch_size = len(batch)
    new_seqs = []
    for i in range(batch_size):
        user_id = batch[i][0]
        seq = copy.deepcopy(batch[i][1][:-1])
        target = batch[i][1][-1]
        
        seq_len = len(seq)
        try:
            replace_pos = np.random.choice(seq_len-1, args.num_replace, replace=False)
        except:
            replace_pos = np.random.choice(seq_len-1, seq_len-1, replace=False)

        for pos in replace_pos:
            synonym_set = synonym_matrix[seq[pos]]
            if len(all_item_embeddings_e5)+1 in synonym_set:
                synonym_set_list = synonym_set.tolist()
                synonym_set_list.remove(len(all_item_embeddings_e5)+1)
                synonym_set = torch.tensor(synonym_set_list).long().to(args.device)

            neighbor_embeddings_e5 = all_item_embeddings_e5[synonym_set-1]
            dp_probs = utility_scores(0.8, all_item_embeddings_e5[seq[pos]-1], neighbor_embeddings_e5)

            smoothed_embedding = torch.sum(neighbor_embeddings_e5 * dp_probs.unsqueeze(1), dim=0)
            dp_synonym = get_similar_item(all_item_embeddings_e5[seq[pos]-1], smoothed_embedding, all_item_embeddings_e5)
            seq[pos] = dp_synonym.item()
        
        new_seqs.append([user_id, seq + [target]])

    return new_seqs

