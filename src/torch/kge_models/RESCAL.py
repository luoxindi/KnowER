import gc
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from .basic_model import BasicModel
from ...py.util.util import to_var


class RESCAL(BasicModel):

    def __init__(self, kgs, args, dim=100):
        super(RESCAL, self).__init__(args, kgs)

        self.dim = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim * self.dim)
        # self.low_values = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        # self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)

    def calc(self, h, t, r):
        # h = F.normalize(h, 2, -1)
        # t = F.normalize(t, 2, -1)
        t = t.view(-1, self.dim, 1)
        r = r.view(-1, self.dim, self.dim)
        tr = torch.matmul(r, t)
        tr = tr.view(-1, self.dim)
        h = h.view(-1, self.dim)
        return -torch.sum(h * tr, -1)

    def get_embeddings(self, h_idx, r_idx, t_idx, mode='entities'):
        h_idx = to_var(h_idx, self.device)
        r_idx = to_var(r_idx, self.device)
        t_idx = to_var(t_idx, self.device)
        b_size = h_idx.shape[0]
        h_emb = self.ent_embeddings(h_idx)
        t_emb = self.ent_embeddings(t_idx)
        r_mat = self.rel_embeddings(r_idx).view(-1, self.dim, self.dim)
        if mode == 'entities':
            candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim)
            candidates = candidates.expand(b_size, self.ent_tot, self.dim)
        else:
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim, self.dim)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim, self.dim)
        return h_emb, r_mat, t_emb, candidates


    def get_score(self, h, r, t):
        b_size = h.shape[0]
    
        if len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 3)
            # this is the head completion case in link prediction
            tr = torch.matmul(r, t.view(b_size, self.dim, 1)).view(b_size, 1, self.dim)
            return -(h * tr).sum(dim=2)
        elif len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 3)
            # this is the tail completion case in link prediction
            hr = torch.matmul(h.view(b_size, 1, self.dim), r).view(b_size, 1, self.dim)
            return -(hr * t).sum(dim=2)
        elif len(r.shape) == 4:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation completion case in link prediction
            h = h.view(b_size, 1, 1, self.dim)
            t = t.view(b_size, 1, self.dim)
            hr = torch.matmul(h, r).view(b_size, self.rel_tot, self.dim)
            return -(hr * t).sum(dim=2)
    
    
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self.calc(h, t, r)
        return score
    
    
    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul
    
    
    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()