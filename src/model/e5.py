import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch

import copy

def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class E5Model(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2', force_download=False)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2', force_download=False)
        self.tokenizer.truncation_side = 'left'
        
        self.device = device

    def init_non_cl(self):
        self.teacher = copy.deepcopy(self.model)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(768, 768, bias=False),
                                        nn.BatchNorm1d(768),
                                        nn.Tanh(), # hidden layer
                                        nn.Linear(768, 768)) # output layer
        
        self.teacher.to(self.device)
        self.predictor.to(self.device)
    
    def forward_target(self, tokens):
        with torch.no_grad():
            outputs = self.teacher(input_ids=tokens[0].to(self.device), 
                                    token_type_ids=tokens[1].to(self.device), 
                                    attention_mask=tokens[2].to(self.device))

            embeddings = average_pool(outputs.last_hidden_state, tokens[2].to(self.device))
            embeddings = F.normalize(embeddings, dim=-1).detach()
    
        return embeddings
    
    def forward_online(self, tokens):
        outputs = self.model(input_ids=tokens[0].to(self.device), 
                            token_type_ids=tokens[1].to(self.device), 
                            attention_mask=tokens[2].to(self.device))

        embeddings = average_pool(outputs.last_hidden_state, tokens[2].to(self.device))
        embeddings = self.predictor(embeddings)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings
    
    def forward_model(self, tokens):
        outputs = self.model(input_ids=tokens[0].to(self.device), 
                            token_type_ids=tokens[1].to(self.device), 
                            attention_mask=tokens[2].to(self.device))

        embeddings = average_pool(outputs.last_hidden_state, tokens[2].to(self.device))
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings
