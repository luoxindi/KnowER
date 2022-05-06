#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.py.util.util import to_tensor_cpu


class PyTorchTrainDataset(Dataset):

    def __init__(self, triples, neg_num, kgs):
        # triples
        self.head = [x[0] for x in triples]
        self.tail = [x[2] for x in triples]
        self.rel = [x[1] for x in triples]
        # total numbers of entities, relations, and triples
        self.neg_num = neg_num
        self.kgs = kgs

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx]

    def collate_fn(self, data):
        batch_h = [item[0] for item in data]
        batch_r = [item[1] for item in data]
        batch_t = [item[2] for item in data]
        batch_neg = self.generate_neg_triples_fast(data, set(self.kgs.relation_triples_list), self.kgs.entities_list, self.neg_num)
        batch_data = list()
        batch_h = to_tensor_cpu(batch_h + [x[0] for x in batch_neg])
        batch_r = to_tensor_cpu(batch_r + [x[1] for x in batch_neg])
        batch_t = to_tensor_cpu(batch_t + [x[2] for x in batch_neg])
        batch_data.append(batch_h)
        batch_data.append(batch_r)
        batch_data.append(batch_t)
        batch_data = torch.stack(batch_data)
        """
        batch_data['batch_h'] = batch_h.squeeze()
        batch_data['batch_t'] = batch_t.squeeze()
        batch_data['batch_r'] = batch_r.squeeze()
        batch_data['batch_y'] = batch_y.squeeze()
        """
        return batch_data

    def generate_neg_triples_fast(self, pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None,
                                  max_try=10):
        if neighbor is None:
            neighbor = dict()
        neg_batch = list()
        for head, relation, tail in pos_batch:
            neg_triples = list()
            nums_to_sample = neg_triples_num
            head_candidates = neighbor.get(head, entities_list)
            tail_candidates = neighbor.get(tail, entities_list)
            for i in range(max_try):
                corrupt_head_prob = np.random.binomial(1, 0.5)
                if corrupt_head_prob:
                    neg_heads = random.sample(head_candidates, nums_to_sample)
                    i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
                else:
                    neg_tails = random.sample(tail_candidates, nums_to_sample)
                    i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
                if i == max_try - 1:
                    neg_triples += list(i_neg_triples)
                    break
                else:
                    i_neg_triples = list(i_neg_triples - all_triples_set)
                    neg_triples += i_neg_triples
                if len(neg_triples) == neg_triples_num:
                    break
                else:
                    nums_to_sample = neg_triples_num - len(neg_triples)
            assert len(neg_triples) == neg_triples_num
            neg_batch.extend(neg_triples)
        assert len(neg_batch) == neg_triples_num * len(pos_batch)
        return neg_batch

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def set_ent_neg_rate(self, rate):
        self.neg_ent = rate

    def set_rel_neg_rate(self, rate):
        self.neg_rel = rate

    def set_bern_flag(self, bern_flag):
        self.bern_flag = bern_flag

    def set_filter_flag(self, filter_flag):
        self.filter_flag = filter_flag

    def get_ent_tot(self):
        return self.ent_total

    def get_rel_tot(self):
        return self.rel_total

    def get_tri_tot(self):
        return self.tri_total


class PyTorchTrainDataLoader(DataLoader):
    def __init__(self, kgs, batch_size, threads, neg_size):
        self.batch_size = batch_size
        self.kgs = kgs
        self.neg_size = neg_size
        self.data = self.__construct_dataset()
        super(PyTorchTrainDataLoader, self).__init__(
            dataset=self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=threads,
            pin_memory=True,
            collate_fn=self.data.collate_fn,
            drop_last=False
        )

    def __construct_dataset(self):
        triples_set = self.kgs.relation_triples_set
        train_dataset = PyTorchTrainDataset(list(triples_set), self.kgs.entities_num,
                                            self.kgs.relations_num, neg_ent=self.neg_size)
        return train_dataset

    def get_ent_tot(self):
        return self.data.get_ent_tot()

    def get_rel_tot(self):
        return self.data.get_rel_tot()

    def get_batch_size(self):
        return self.batch_size

    """interfaces to set essential parameters"""

    def set_sampling_mode(self, sampling_mode):
        self.dataset.set_sampling_mode(sampling_mode)

    def set_work_threads(self, work_threads):
        self.num_workers = work_threads

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches
        self.batch_size = self.tripleTotal // self.nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.dataset.set_ent_neg_rate(rate)

    def set_rel_neg_rate(self, rate):
        self.dataset.set_rel_neg_rate(rate)

    def set_bern_flag(self, bern_flag):
        self.dataset.set_bern_flag(bern_flag)

    def set_filter_flag(self, filter_flag):
        self.dataset.set_filter_flag(filter_flag)

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.dataset.get_ent_tot()

    def get_rel_tot(self):
        return self.dataset.get_rel_tot()

    def get_triple_tot(self):
        return self.dataset.get_tri_tot()
