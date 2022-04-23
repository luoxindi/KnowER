import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.py.evaluation.evaluation import valid, test
from src.py.util.util import to_tensor
from src.torch.kge_models.basic_model import BasicModel


class ConnectE(BasicModel):

    def __init__(self, args, kgs):
        super(ConnectE, self).__init__(args, kgs)
        self.args = args
        self.kgs = kgs
        # self.out_folder = r'D:\OPENEA-pytorch\result'

    def init(self):
        self._define_variables()
        # customize parameters
        #assert self.args.init == 'unit'
        assert self.args.alignment_module == 'mapping'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.ent_l2_norm is True

        assert self.args.alpha > 1

    def _define_variables(self):
        print(self.device)
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, 200)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, 200)
        self.type_embeds = nn.Embedding(3851, self.args.dim)
        self.mapping_matrix = nn.Parameter(torch.Tensor(200, 100))
        # nn.init.orthogonal_(self.mapping_matrix, gain=1)
        self.eye_mat = torch.Tensor(self.args.dim, self.args.dim).to(self.device)
        nn.init.eye_(self.eye_mat)

        self.margin = torch.Tensor([2]).to(self.device)
        print(self.margin)
        if self.args.init == 'xavier':
            nn.init.xavier_uniform_(self.ent_embeds.weight.data)
            nn.init.xavier_uniform_(self.rel_embeds.weight.data)
            nn.init.xavier_uniform_(self.type_embeds.weight.data)
        elif self.args.init == 'normal':
            std = 1.0 / math.sqrt(self.args.dim)
            nn.init.normal_(self.ent_embeds.weight.data, 0, std)
            nn.init.normal_(self.rel_embeds.weight.data, 0, std)
            nn.init.normal_(self.type_embeds.weight.data, 0, std)
        elif self.args.init == 'uniform':
            bnd = math.sqrt(6) / math.sqrt(self.ent_embeds.shape[0] + self.ent_embeds.shape[1])
            nn.init.uniform_(self.ent_embeds.weight.data, -bnd, bnd)
            bnd = math.sqrt(6) / math.sqrt(self.rel_embeds.shape[0] + self.rel_embeds.shape[1])
            nn.init.uniform_(self.rel_embeds.weight.data, -bnd, bnd)
            bnd = math.sqrt(6) / math.sqrt(self.type_embeds.shape[0] + self.type_embeds.shape[1])
            nn.init.uniform_(self.type_embeds.weight.data, -bnd, bnd)
            bnd = math.sqrt(6) / math.sqrt(self.mapping_matrix.shape[0] + self.mapping_matrix.shape[1])
            nn.init.uniform_(self.mapping_matrix, -bnd, bnd)
        self.type_embeds.weight.data = F.normalize(self.type_embeds.weight.data, 2, -1)
        self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)
        self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)

    def generate_transE_loss(self, data):
        ph = data['pos_hs']
        pr = data['pos_rs']
        pt = data['pos_ts']
        nh = data['neg_hs']
        nr = data['neg_rs']
        nt = data['neg_ts']
        batch_size_now = ph.shape[0]
        '''ph = F.normalize(self.ent_embeds(ph), 2, -1)
        pr = F.normalize(self.rel_embeds(pr), 2, -1)
        pt = F.normalize(self.ent_embeds(pt), 2, -1)
        nh = F.normalize(self.ent_embeds(nh), 2, -1)
        nr = F.normalize(self.rel_embeds(nr), 2, -1)
        nt = F.normalize(self.ent_embeds(nt), 2, -1)'''
        ph = self.ent_embeds(ph)
        pr = self.rel_embeds(pr)
        pt = self.ent_embeds(pt)
        nh = self.ent_embeds(nh)
        nr = self.rel_embeds(nr)
        nt = self.ent_embeds(nt)
        if self.args.loss_norm == "L2":
            pos = torch.pow(torch.norm(ph + pr - pt, 2, -1), 2)
            neg = torch.pow(torch.norm(nh + nr - nt, 2, -1), 2)
        else:
            pos = torch.norm(ph + pr - pt, 1, -1)
            neg = torch.norm(nh + nr - nt, 1, -1)
        pos = pos.view(batch_size_now, -1)
        neg = neg.view(batch_size_now, -1)
        return torch.sum(torch.relu_(pos - neg + self.margin))

    def generate_mapping_loss(self, data):
        ent = data['ent']
        etype = data['type']
        ent2 = data['ent2']
        type2 = data['type2']
        ent = self.ent_embeds(ent)
        etype = self.type_embeds(etype)
        seed_mapped_entity = torch.matmul(ent, self.mapping_matrix)
        distance = seed_mapped_entity - etype
        align_loss = torch.pow(torch.norm(distance, 2, -1), 2)

        ent = self.ent_embeds(ent2)
        etype = self.type_embeds(type2)
        seed_mapped_entity = torch.matmul(ent, self.mapping_matrix)
        distance = seed_mapped_entity - etype
        align_loss2 = torch.pow(torch.norm(distance, 2, -1), 2)
        """
        orthogonal_loss = torch.mean(
            torch.sum(torch.pow(torch.matmul(self.mapping_matrix, self.mapping_matrix.t()) - self.eye_mat, 2), -1)
        )
        """
        return torch.mean(torch.relu_(align_loss - align_loss2 + self.margin))

    def valid(self, stop_metric):
        """
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_et_ent, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.type_embeds(to_tensor(self.kgs.valid_et_type, self.device)), 2, -1)
        """
        seed_entity1 = self.ent_embeds(to_tensor(self.kgs.valid_et_ent, self.device))
        seed_entity2 = self.type_embeds(to_tensor(self.kgs.valid_et_type, self.device))
        hits1_12, mrr_12 = valid(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), self.mapping_matrix.cpu().detach().numpy(),
                                 self.args.top_k,self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def test(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1, self.device)), 2, -1)
        seed_entity2 = F.normalize(self.type_embeds(to_tensor(entities2, self.device)), 2, -1)
        _, _, _, sim_list = test(seed_entity1.cpu().detach().numpy(), seed_entity2.cpu().detach().numpy(), self.mapping_matrix.cpu().detach().numpy(),
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list
