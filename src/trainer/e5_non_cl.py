from .utils import *
from .base import *
import torch
import torch.nn.functional as F

import numpy as np
from abc import *
import pdb

def get_input_seq_template(args, seq, meta):
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new', 'sports', 'office']:
        prompt = "query: "
        for item in seq:
            try:
                title = meta[item][0]
            except:
                import pdb; pdb.set_trace()
            prompt += "'"
            prompt += title
            prompt += "', a product from "

            category = meta[item][2]
            category = ', '.join(category.split(', ')[-2:])

            prompt += category
            prompt += ' category. \n'

    return prompt

class E5Trainer_non_cl(BaseTrainer):
    def __init__(self, args, model, meta, train_loader, val_loader, DP_loader):
        super().__init__(args, model, train_loader, val_loader, DP_loader)
        # get dataloader
        self.meta = meta

        self.ce = torch.nn.CrossEntropyLoss()
        self.mr = torch.nn.MarginRankingLoss(self.args.margin)
    
    def calculate_loss(self, batch):
        user_id, seq, purchase_tokens, input_text, target_text, batch_size = batch

        input_text_dp = self.prepare_DP_samples(user_id)
        input_tokens_all = self.prepare_tokens(input_text, input_text_dp)

        # calculate user embeddings
        p_all, z_all = self.calculate_embeddings(input_tokens_all)
        
        # compute alignment loss
        loss_a = self.get_alignment_loss(p_all[:batch_size], z_all[batch_size:]) + \
                    self.get_alignment_loss(p_all[batch_size:], z_all[:batch_size])
        
        # compute uniformity loss
        loss_u = self.get_uniformity_loss(p_all[:batch_size], z_all[batch_size:]) + \
                    self.get_uniformity_loss(p_all[batch_size:], z_all[:batch_size])
        
        # calculate target item uniformity loss
        loss_s = self.get_target_item_uniformity_loss(seq, target_text, batch_size)

        loss = self.get_contrastive_loss_purchase(purchase_tokens, batch_size) \
                    + self.args.mce_gamma_a * loss_a + self.args.mce_gamma_u * loss_u \
                    + self.args.mce_gamma_s * loss_s

        return loss
    
    def calculate_embeddings(self, input_tokens_all):        
        p_all = self.model.forward_target(input_tokens_all) # no grad
        z_all = self.model.forward_online(input_tokens_all)
        return p_all, z_all
    
    def get_contrastive_loss_purchase(self, purchases_contrastive, batch_size):
        embeddings = self.model.forward_model(purchases_contrastive).reshape(batch_size, (1 + 1 + self.args.num_non_purchase), -1)
        
        # cosine between positive pairs
        positive_logit = torch.sum(embeddings[:, 0] * embeddings[:, 1], dim=-1, keepdim=True)
        # cosine between negative pairs
        negative_logit = torch.sum(embeddings[:, 0].unsqueeze(1) * embeddings[:, 2:], dim=-1)

        logits = torch.cat([negative_logit, positive_logit], dim=-1) / 0.01
    
        labels = torch.ones(len(logits)).long().to(self.local_device) * (logits.shape[-1] - 1) 

        # cross entropy
        loss = self.ce(logits, labels)
        
        return loss
    
    def get_target_item_uniformity_loss(self, seq, target_text, batch_size):
        # compute item_level uniformity loss
        target_id = [x[-1] for x in seq]
        target_synonym = self.synonym_matrix[target_id]
        # loss_scl = 
        target_aug_embedding = []
        for i in range(batch_size):
            neighbor_embeddings_e5 = self.candidate_embeddings_e5[target_synonym[i]-1]
            dp_probs = utility_scores(0.8, self.candidate_embeddings_e5[target_id[i]-1], neighbor_embeddings_e5)
            smoothed_target_embedding = torch.sum(neighbor_embeddings_e5 * dp_probs.unsqueeze(1), dim=0)
            target_aug_embedding.append(smoothed_target_embedding) # no grad
        
        target_aug_embedding = torch.stack(target_aug_embedding)
        target_aug_embedding = F.normalize(target_aug_embedding, dim=-1)

        target_tokens = self.prepare_target_tokens(target_text)
        target_embedding = self.model.forward_model(target_tokens)
        
        loss_scl = self.get_uniformity_loss(target_embedding, target_aug_embedding)

        return loss_scl

    def get_alignment_loss(self, p, z):
        m = z.shape[0]
        n = z.shape[1]
        # print(m, n)
        J_m = self.centering_matrix(m).detach().to(z.device)

        P = (1. / m) * (p.T @ J_m @ p) + self.args.mce_mu * torch.eye(n).to(z.device)
        Q = (1. / m) * (z.T @ J_m @ z) + self.args.mce_mu * torch.eye(n).to(z.device)
        
        return torch.trace(- P @ self.matrix_log(Q, self.args.mce_order))

    def get_uniformity_loss(self, p, z):
        m = z.shape[0]
        n = z.shape[1]
        # print(m, n)
        J_m = self.centering_matrix(m).detach().to(z.device)

        P = self.args.mce_lamda * torch.eye(n).to(z.device)
        Q = (1. / m) * (p.T @ J_m @ z) + self.args.mce_mu * torch.eye(n).to(z.device)

        return torch.trace(- P @ self.matrix_log(Q, self.args.mce_order))
    
    def centering_matrix(self, m):
        J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
        return J_m

    # Taylor expansion
    def matrix_log(self, Q, order=4):
        n = Q.shape[0]
        Q = Q - torch.eye(n).detach().to(Q.device)
        cur = Q
        res = torch.zeros_like(Q).detach().to(Q.device)
        for k in range(1, order + 1):
            if k % 2 == 1:
                res = res + cur * (1. / float(k))
            else:
                res = res - cur * (1. / float(k))
            cur = cur @ Q

        return res

    def calculate_metrics(self, batch):
        batch_size = batch[-1]
        embeddings = self.model.forward_model(batch)
        
        embeddings = embeddings.reshape(batch_size, (1 + 1 + self.args.num_non_purchase), -1)

        # cosine between positive pairs
        positive_logit = torch.sum(embeddings[:, 0] * embeddings[:, 1], dim=-1, keepdim=True)
        
        # cosine between negative pairs
        negative_logit = torch.sum(embeddings[:, 0].unsqueeze(1) * embeddings[:, 2:], dim=-1)

        scores = torch.cat([negative_logit, positive_logit], dim=-1) / 0.01
        labels = torch.ones(len(scores)).long().to(self.local_device) * (scores.shape[-1] - 1)

        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)

        return metrics

    def prepare_DP_samples(self, user_id):
        input_text_dp = []
        for uid in user_id:
            dp_seq = self.dp_seqs[uid] 
            dp_text = get_input_seq_template(self.args, dp_seq, self.meta)
            input_text_dp.append(dp_text)

        return input_text_dp

    def prepare_tokens(self, input_text, input_text_dp):
        text_all = input_text + input_text_dp
        input_tokens_dp = self.model.tokenizer(text_all, max_length=256, truncation=True, padding=True, return_tensors="pt")
        
        return (input_tokens_dp['input_ids'], input_tokens_dp['token_type_ids'], input_tokens_dp['attention_mask'])

    def prepare_target_tokens(self, target_text):
        target_tokens = self.model.tokenizer(target_text, max_length=256, truncation=True, padding=True, return_tensors="pt")
        return (target_tokens['input_ids'], target_tokens['token_type_ids'], target_tokens['attention_mask'])