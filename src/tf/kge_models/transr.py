import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, empty
from torch.cuda import empty_cache
from torch.nn import Parameter
from tqdm import tqdm

from .basic_model import BasicModel
from ...py.load import read
from ...py.util.util import to_var


class TransR(BasicModel):

    def __init__(self, kgs, args, dim_e=100, dim_r=100, p_norm=1, norm_flag=True, rand_init=False, margin=None):
        super(TransR, self).__init__(args, kgs)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.projected = False
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.rand_init = rand_init
        self.projected_entities = Parameter(empty(size=(self.rel_tot,
                                                        self.ent_tot,
                                                        self.dim_r)),
                                            requires_grad=False)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)
        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_r)
            for i in range(min(self.dim_e, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_e)
            for i in range(self.rel_tot):
                self.transfer_matrix.weight.data[i] = identity
        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def calc(self, h, t, r):
        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1)
        return score

    def transfer(self, e, r_transfer):
        proj_e = torch.matmul(e.view(-1, 1, self.dim_e), r_transfer)
        return proj_e.view(-1, self.dim_r)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.transfer_matrix(batch_r).view(-1, self.dim_e, self.dim_r)
        h = self.transfer(h, r_transfer)
        t = self.transfer(t, r_transfer)
        score = self.calc(h, t, r).flatten()
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def get_embeddings(self, hid, rid, tid, mode='entity'):
        h = to_var(hid, self.device)
        r = to_var(rid, self.device)
        t = to_var(tid, self.device)
        self.evaluate_projections()
        r_embs = self.rel_embeddings(r)

        if mode == 'entity':
            proj_h = self.projected_entities[rid, hid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            proj_t = self.projected_entities[rid, tid].unsqueeze(1)  # shape: (b_size, 1, emb_dim)
            candidates = self.projected_entities[rid]  # shape: (b_size, self.n_rel, self.emb_dim)
            return proj_h, r_embs, proj_t, candidates
        else:
            proj_h = self.projected_entities[:, hid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            b_size = proj_h.shape[0]
            proj_t = self.projected_entities[:, tid].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim_r)
            candidates = candidates.expand(b_size, self.rel_tot, self.dim_r)
            return proj_h, r_embs, proj_t, candidates

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.transfer_matrix(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(r_transfer ** 2)) / 4
        return regul * regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def save(self):
        ent_embeds = self.ent_embeddings.cpu().weight.data
        rel_embeds = self.rel_embeddings.cpu().weight.data
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        transfer_matrix = self.transfer_matrix.cpu().weight.data
        read.save_special_embeddings(self.out_folder, 'transfer_matrix', '', transfer_matrix, None)

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.projected:
            return
        for i in tqdm(range(self.ent_tot), unit='entities', desc='Projecting entities'):

            mask = i
            '''if projection_matrices.is_cuda:
                empty_cache()'''
            projection_matrices = self.transfer_matrix.weight.data
            projection_matrices = projection_matrices.view(self.rel_tot, self.dim_e, self.dim_r)

            ent = self.ent_embeddings.weight[mask]
            proj_ent = torch.matmul(ent.view(1, self.dim_e), projection_matrices)
            if projection_matrices.is_cuda:
                empty_cache()
            # proj_ent = proj_ent.view(self.rel_tot, self.dim_r, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.rel_tot, self.dim_r)

            del proj_ent, projection_matrices
            #if i % 100 == 0:
            #gc.collect()

        self.projected = True
