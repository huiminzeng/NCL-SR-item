from model import *
from config import *
from .utils import *

import sys
sys.path.append('../DP')
from DP import *

import copy 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from abc import ABCMeta

import pdb

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader=None, val_loader=None, DP_loader=None):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.DP_loader = DP_loader
        
        if args.ds_mode == '1_gpu':
            self.local_device = args.device
            self.model = model.to(self.local_device)

            # get optimizer
            self.optimizer = self._create_optimizer()

            if args.enable_lr_schedule:
                if args.enable_lr_warmup:
                    self.lr_scheduler = self.get_linear_schedule_with_warmup(
                        self.optimizer, args.warmup_steps, len(self.train_loader) * self.num_epochs)
                else:
                    self.lr_scheduler = optim.lr_scheduler.StepLR(
                        self.optimizer, step_size=args.decay_step, gamma=args.gamma)
                    
        self.local_epochs = args.epochs
        self.metric_ks = args.metric_ks

    def train(self):
        self.best_metric = -1e3
        self.best_meter = None
        # self.validate()

        for epoch in range(self.args.epochs):
            if self.args.ds_mode == '1_gpu':
                self.generate_dp_augmentations()
                self.train_one_epoch(epoch)

                break

        # if is_rank_0():
        #     for k in [5,10]:
        #         recall = self.best_meter.averages()['Recall@%d' % k]
        #         ndcg = self.best_meter.averages()['NDCG@%d' % k]

        #         print('Recall@{}: {:.4f}'.format(k, recall))
        #         print('NDCG@{}: {:.4f}'.format(k, ndcg))

        #         print("="*50)

        if is_rank_0():
            print("Training is over.")  
      
    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)
        
        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.model.train()

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
    
            self.clip_gradients(self.args.max_grad_norm)
            self.optimizer.step()

            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            # momentum update of the parameters of the teacher network
            with torch.no_grad():
                for param_q, param_k in zip(self.model.model.parameters(), self.model.teacher.parameters()):
                    param_k.data.mul_(self.args.teacher_momentum).add_((1 - self.args.teacher_momentum) * param_q.detach().data)

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                    'Epoch {}, Batch [{}]/[{}]loss {:.3f} '.format(epoch+1, batch_idx, len(self.train_loader), average_meter_set['loss'].avg)) 

            # break

            # if batch_idx % self.args.print_freq == 0 and batch_idx != 0:
            self.validate() 
            
            break
      
    def validate(self):
        model_save_name = os.path.join(self.args.export_dir, 'model.checkpoint')
        model_checkpoint = {'state_dict': self.model.state_dict()}
        torch.save(model_checkpoint, model_save_name)
        # self.model.eval()
        # average_meter_set = AverageMeterSet()
        # with torch.no_grad():
        #     tqdm_dataloader = tqdm(self.val_loader)
        #     for batch_idx, batch in enumerate(tqdm_dataloader):
        #         metrics = self.calculate_metrics(batch)
        #         self._update_meter_set(average_meter_set, metrics)
        #         self._update_dataloader_metrics(tqdm_dataloader, average_meter_set)

        # for k in [5,10]:
        #     recall = average_meter_set.averages()['Recall@%d' % k]
        #     ndcg = average_meter_set.averages()['NDCG@%d' % k]

        #     print('Recall@{}: {:.4f}'.format(k, recall))
        #     print('NDCG@{}: {:.4f}'.format(k, ndcg))

        #     print("="*50)

        # current_metric = average_meter_set.averages()["NDCG@10"] + average_meter_set.averages()["Recall@10"]

        # if current_metric > self.best_metric:
        #     self.best_metric = current_metric
        #     self.best_meter = average_meter_set

        #     if self.args.ds_mode == '1_gpu':
        #         model_save_name = os.path.join(self.args.export_dir, 'model.checkpoint')
        #         model_checkpoint = {'state_dict': self.model.state_dict()}
        #         torch.save(model_checkpoint, model_save_name)
        
        
    def to_device(self, batch):
        return [x.to(self.device) for x in batch]

    # @abstractmethod
    def calculate_loss(self, batch):
        pass
    
    # @abstractmethod
    def calculate_metrics(self, batch):
        pass
    
    def clip_gradients(self, limit=1.0):
        nn.utils.clip_grad_norm_(self.model.parameters(), limit)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + ', '.join(s + ' {:.4f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(*(meter_set[k].avg for k in description_metrics))
        # if is_rank_0():
        #     tqdm_dataloader.set_description(description)
    
    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    def get_parameters(self, model):
        args = self.args
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]

        return optimizer_grouped_parameters
        
    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def generate_dp_augmentations(self):
        self.model.eval()
        self.dp_seqs = {}
        print('****************** Generating DP samples ******************')
        with torch.no_grad():
            candidate_embeddings_e5 = calculate_all_item_embeddings(self.model, self.meta, self.args)
            synonym_matrix, synonym_dict_1_order = build_1_order_synonym_dict(0.95, candidate_embeddings_e5)
            synonym_matrix = build_2_order_synonym_dict(synonym_matrix, synonym_dict_1_order)
            
            self.synonym_matrix = synonym_matrix
            self.candidate_embeddings_e5 = candidate_embeddings_e5
            
            tqdm_dataloader = tqdm(self.DP_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch_size = len(batch)
                new_seqs = calculate_substitutions(batch, synonym_matrix, candidate_embeddings_e5, self.args)
                for i in range(batch_size):
                    user_id, dp_seq = new_seqs[i]
                    self.dp_seqs[user_id] = dp_seq
    
    
