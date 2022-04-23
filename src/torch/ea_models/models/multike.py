import math
import multiprocessing as mp
import os
import time

import numpy as np
import random
import gc
# from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import src.trainer.batch as bat
from src.approaches.attre import early_stop
from src.approaches.literal_encoder import LiteralEncoder
from src.approaches.predicate_align import PredicateAlignModel
from src.modules.finding.evaluation import valid, test
from src.modules.load.read import generate_sup_attribute_triples
from src.trainer.batch import task_divide
from src.trainer.util import to_tensor


'''def test_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.test_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag test results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12'''


'''def _compute_weight(embeds1, embeds2, embeds3):
    def min_max_normalization(mat):
        min_ = np.min(mat)
        max_ = np.max(mat)
        return (mat - min_) / (max_ - min_)

    other_embeds = (embeds1 + embeds2 + embeds3) / 3
    other_embeds = preprocessing.normalize(other_embeds)
    embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, other_embeds.T)
    weights = np.diag(sim_mat)
    return np.mean(weights)'''


'''def wva(embeds1, embeds2, embeds3):
    weight1 = _compute_weight(embeds1, embeds2, embeds3)
    weight2 = _compute_weight(embeds2, embeds1, embeds3)
    weight3 = _compute_weight(embeds3, embeds1, embeds2)
    return weight1, weight2, weight3
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight
    print('final weights', weight1, weight2, weight3)
    ent_embeds = weight1 * embeds1 + \
                 weight2 * embeds2 + \
                 weight3 * embeds3
    return ent_embeds


def valid_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.valid_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.valid_entities2 + model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag valid results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)

    del nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1
    del nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2
    del embeds1, embeds2
    gc.collect()

    return mrr_12'''


def valid_temp(model, embed_choice='avg', w=(1, 1, 1)):
    if embed_choice == 'nv':
        ent_embeds = model.name_embeds.detach()
    elif embed_choice == 'rv':
        ent_embeds = model.rv_ent_embeds.weight.data
    elif embed_choice == 'av':
        ent_embeds = model.av_ent_embeds.weight.data
    elif embed_choice == 'final':
        ent_embeds = model.ent_embeds.weight.data
    elif embed_choice == 'avg':
        ent_embeds = w[0] * model.name_embeds.detach() + \
                     w[1] * model.rv_ent_embeds.detach() + \
                     w[2] * model.av_ent_embeds.weight.detach()
    else:  # 'final'
        ent_embeds = model.ent_embeds.detach()
    print(embed_choice, 'valid results:')
    embeds1 = F.normalize(ent_embeds[model.kgs.valid_entities1, ], 2, -1).numpy()
    embeds2 = F.normalize(ent_embeds[model.kgs.valid_entities2 + model.kgs.test_entities2,], 2, -1).numpy()
    hits1_12, mrr_12 = valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                             normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def read_word2vec(file_path, vector_dimension):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, attribute, value, w in pos_batch:
        for i in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value, w) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value, w))
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_attribute_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2, entity_list1,
                                          entity_list2, batch_size, steps, out_queue, neighbor1, neighbor2,
                                          neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                               entity_list1, entity_list2, batch_size,
                                                               step, neighbor1, neighbor2, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def logistic_loss(phs, prs, pts, nhs, nrs, nts, loss_norm):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts
    if loss_norm == 'L1':  # L1 score
        pos_score = torch.sum(torch.abs(pos_distance), dim=1)
        neg_score = torch.sum(torch.abs(neg_distance), dim=1)
    else:  # L2 score
        pos_score = torch.norm(pos_distance, 2, -1)
        neg_score = torch.norm(neg_distance, 2, -1)
    pos_loss = torch.sum(torch.log(1 + torch.exp(pos_score)))
    neg_loss = torch.sum(torch.log(1 + torch.exp(-neg_score)))
    loss = pos_loss + neg_loss
    return loss


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = bat.generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = bat.generate_pos_triples(triple_list2, batch_size2, step)
    neg_batch1 = generate_neg_attribute_triples(pos_batch1, triple_set1, entity_list1,
                                                neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_attribute_triples(pos_batch2, triple_set2, entity_list2,
                                                neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2


def positive_loss_with_weight(phs, pas, pvs, pws):
    pos_distance = phs + pas - pvs
    pos_score = -torch.norm(pos_distance)
    pos_score = torch.log(1 + torch.exp(-pos_score))
    pos_score = torch.multiply(pos_score, pws)
    pos_loss = torch.sum(pos_score)
    return pos_loss


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = torch.sum(torch.norm(distance, 2, -1))
    return loss


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye, orthogonal_weight, norm_w=0.0001):
    mapped_ents2 = torch.matmul(view_embeds, mapping)
    mapped_ents2 = F.normalize(mapped_ents2, 2, -1)
    map_loss = torch.sum(torch.norm(shared_embeds - mapped_ents2))
    norm_loss = torch.sum(torch.norm(mapping))
    orthogonal_loss = torch.sum(torch.norm(torch.matmul(mapping, mapping.t()) - eye), 1)
    return map_loss + orthogonal_weight * orthogonal_loss + norm_w * norm_loss


class MultiKE(nn.Module):

    def __init__(self, args, kgs):
        super(MultiKE, self).__init__()
        self.entity_local_name_dict = None
        self.predicate_align_model = None
        self.entities = None
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = -1
        self.args = args
        self.kgs = kgs
        self.batch_norm = torch.nn.BatchNorm2d(num_features=self.args.dim, dtype=torch.float32)
        self.dense = nn.Linear(400, self.args.dim)
        self.conv1 = nn.Conv2d(1, 2, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(2, 2, (1, 1), (1, 1))

    def init(self):
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        self.entity_local_name_dict = self._get_local_name_by_name_triple()
        print('len(self.entity_local_name_dict):', len(self.entity_local_name_dict))
        self._generate_literal_vectors()
        self._generate_name_vectors_mat()
        self._generate_attribute_value_vectors()
        self.predicate_align_model = PredicateAlignModel(self.kgs, self.args)
        self._define_variables()

    def _get_local_name_by_name_triple(self, name_attribute_list=None):
        if name_attribute_list is None:
            if 'D_Y' in self.args.training_data:
                name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
            elif 'D_W' in self.args.training_data:
                name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
            else:
                name_attribute_list = {}

        triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        id_ent_dict = {}
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e
        print(len(id_ent_dict))

        name_ids = set()
        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)

        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        print(name_ids)

        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        print(len(ents))
        for (e, a, v) in triples:
            if a in name_ids:
                local_name_dict[id_ent_dict[e]] = v
        print('after name_ids:', len(local_name_dict))
        for e in ents:
            if id_ent_dict[e] not in local_name_dict:
                    local_name_dict[id_ent_dict[e]] = id_ent_dict[e].split('/')[-1].replace('_', ' ')
        return local_name_dict

    def _generate_literal_vectors(self):
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        value_list = [v for (_, _, v) in cleaned_attribute_triples_list1 + cleaned_attribute_triples_list2]
        local_name_list = list(self.entity_local_name_dict.values())
        self.literal_list = list(set(value_list + local_name_list))
        print('literal num:', len(local_name_list), len(value_list), len(self.literal_list))

        if os.path.exists('../../literal_embeddings.npy'):
            self.literal_vectors_mat = np.load('../../literal_embeddings.npy')
        else:
            word2vec = read_word2vec(self.args.word2vec_path, self.args.word2vec_dim)
            literal_encoder = LiteralEncoder(self.literal_list, word2vec, self.args, 300)
            self.literal_vectors_mat = literal_encoder.encoded_literal_vector
        assert self.literal_vectors_mat.shape[0] == len(self.literal_list)
        self.literal_id_dic = dict()
        for i in range(len(self.literal_list)):
            self.literal_id_dic[self.literal_list[i]] = i
        assert len(self.literal_list) == len(self.literal_id_dic)

    def _generate_name_vectors_mat(self):
        name_ordered_list = list()
        num = len(self.entities)
        print("total entities:", num)
        entity_id_uris_dic = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))
        entity_id_uris_dic2 = dict(zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))
        entity_id_uris_dic.update(entity_id_uris_dic2)
        print('total entities ids:', len(entity_id_uris_dic))
        assert len(entity_id_uris_dic) == num
        for i in range(num):
            assert i in entity_id_uris_dic
            entity_uri = entity_id_uris_dic.get(i)
            assert entity_uri in self.entity_local_name_dict
            entity_name = self.entity_local_name_dict.get(entity_uri)
            entity_name_index = self.literal_id_dic.get(entity_name)
            name_ordered_list.append(entity_name_index)
        print('name_ordered_list', len(name_ordered_list))
        name_mat = to_tensor(self.literal_vectors_mat[name_ordered_list,])
        print("entity name embeddings mat:", type(name_mat), name_mat.shape)
        if self.args.literal_normalize:
            name_mat = F.normalize(name_mat, 2, -1)
        self.local_name_vectors = name_mat

    def _generate_attribute_value_vectors(self):
        self.literal_set = set(self.literal_list)
        values_set = set()
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        attribute_triples_list1, attribute_triples_list2 = set(), set()
        for h, a, v in cleaned_attribute_triples_list1:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list1.add((h, a, v))

        for h, a, v in cleaned_attribute_triples_list2:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list2.add((h, a, v))
        print("selected attribute triples", len(attribute_triples_list1), len(attribute_triples_list2))
        values_id_dic = dict()
        values_list = list(values_set)
        num = len(values_list)
        for i in range(num):
            values_id_dic[values_list[i]] = i
        id_attribute_triples1 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list1])
        id_attribute_triples2 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list2])
        self.kgs.kg1.set_attributes(id_attribute_triples1)
        self.kgs.kg2.set_attributes(id_attribute_triples2)
        sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.kgs.train_links, self.kgs.kg1.av_dict,
                                                                    self.kgs.kg2.av_dict)
        self.kgs.kg1.add_sup_attribute_triples(sup_triples1)
        self.kgs.kg2.add_sup_attribute_triples(sup_triples2)
        num = len(values_id_dic)
        value_ordered_list = list()
        for i in range(num):
            value = values_list[i]
            value_index = self.literal_id_dic.get(value)
            value_ordered_list.append(value_index)
        print('value_ordered_list', len(value_ordered_list))
        value_vectors = to_tensor(self.literal_vectors_mat[value_ordered_list,])
        print("value embeddings mat:", type(value_vectors), value_vectors.shape)
        if self.args.literal_normalize:
            value_vectors = F.normalize(value_vectors, 2, -1)
        self.value_vectors = value_vectors

    def _define_variables(self):
        self.rv_ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.av_ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.rel_embeds = nn.Embedding(self.kgs.relations_num, self.args.dim)
        self.attr_embeds = nn.Embedding(self.kgs.attributes_num, self.args.dim)
        self.ent_embeds = nn.Embedding(self.kgs.entities_num, self.args.dim)
        self.literal_embeds = F.normalize(self.value_vectors, 2, -1)
        nn.init.xavier_uniform_(self.rv_ent_embeds.weight.data)
        nn.init.xavier_uniform_(self.rel_embeds.weight.data)
        nn.init.xavier_uniform_(self.av_ent_embeds.weight.data)
        nn.init.xavier_uniform_(self.attr_embeds.weight.data)
        nn.init.xavier_uniform_(self.ent_embeds.weight.data)
        self.name_embeds = F.normalize(self.local_name_vectors, 2, -1)
        self.margin = nn.Parameter(torch.Tensor([1.5]))
        self.margin.requires_grad = False
        '''self.rv_ent_embeds.weight.data = F.normalize(self.rv_ent_embeds.weight.data, 2, -1)
        self.rel_embeds.weight.data = F.normalize(self.rel_embeds.weight.data, 2, -1)
        self.av_ent_embeds.weight.data = F.normalize(self.av_ent_embeds.weight.data, 2, -1)
        self.attr_embeds.weight.data = F.normalize(self.attr_embeds.weight.data, 2, -1)
        self.ent_embeds.weight.data = F.normalize(self.ent_embeds.weight.data, 2, -1)'''

        self.nv_mapping = Variable(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.nv_mapping, gain=1)
        self.rv_mapping = Variable(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.rv_mapping, gain=1)
        self.av_mapping = Variable(torch.Tensor(self.args.dim, self.args.dim), requires_grad=True)
        nn.init.orthogonal_(self.av_mapping, gain=1)
        self.eye_mat = torch.Tensor(self.args.dim, self.args.dim)
        nn.init.eye_(self.eye_mat)

    # --- The followings are view-specific embedding models --- #

    def define_relation_view_graph(self, data):
        rel_pos_hs = data['rel_pos_hs']
        rel_pos_rs = data['rel_pos_rs']
        rel_pos_ts = data['rel_pos_ts']
        rel_neg_hs = data['rel_neg_hs']
        rel_neg_rs = data['rel_neg_rs']
        rel_neg_ts = data['rel_neg_ts']
        batch_size = rel_pos_hs.shape[0]
        rel_phs = F.normalize(self.rv_ent_embeds(rel_pos_hs), 2, -1)
        rel_prs = F.normalize(self.rel_embeds(rel_pos_rs), 2, -1)
        rel_pts = F.normalize(self.rv_ent_embeds(rel_pos_ts), 2, -1)
        rel_nhs = F.normalize(self.rv_ent_embeds(rel_neg_hs), 2, -1)
        rel_nrs = F.normalize(self.rel_embeds(rel_neg_rs), 2, -1)
        rel_nts = F.normalize(self.rv_ent_embeds(rel_neg_ts), 2, -1)
        # print("relation cv")
        name_phs = F.normalize(self.name_embeds[rel_pos_hs.to(torch.long), :], 2, -1)
        name_pts = F.normalize(self.name_embeds[rel_pos_ts.to(torch.long), :], 2, -1)
        
        final_phs = F.normalize(self.ent_embeds(rel_pos_hs), 2, -1)
        final_pts = F.normalize(self.ent_embeds(rel_pos_ts), 2, -1)
        '''rel_phs = self.rv_ent_embeds(rel_pos_hs)
        rel_prs = self.rel_embeds(rel_pos_rs)
        rel_pts = self.rv_ent_embeds(rel_pos_ts)
        rel_nhs = self.rv_ent_embeds(rel_neg_hs)
        rel_nrs = self.rel_embeds(rel_neg_rs)
        rel_nts = self.rv_ent_embeds(rel_neg_ts)
        # print("relation cv")
        name_phs = self.name_embeds[rel_pos_hs.to(torch.long), :]
        name_pts = self.name_embeds[rel_pos_ts.to(torch.long), :]
        final_phs = self.ent_embeds(rel_pos_hs)
        final_pts = self.ent_embeds(rel_pos_ts)'''
        relation_loss = logistic_loss(rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts, 'L2')
        # print("relation cv")
        pos = torch.norm(final_phs + rel_prs - rel_pts, 2, -1)
        align_loss = torch.sum(pos)
        pos = torch.norm(rel_phs + rel_prs - final_pts, 2, -1)
        align_loss += torch.sum(pos)
        align_loss += 0.5 * alignment_loss(final_phs, name_phs)
        align_loss += 0.5 * alignment_loss(final_pts, name_pts)
        relation_loss += align_loss
        return relation_loss

    def define_attribute_view_graph(self, data):
        attr_pos_hs = data['attr_pos_hs']
        attr_pos_as = data['attr_pos_as']
        attr_pos_vs = data['attr_pos_vs']
        attr_pos_ws = data['attr_pos_ws']

        attr_phs = F.normalize(self.av_ent_embeds(attr_pos_hs), 2, -1)
        attr_pas = F.normalize(self.attr_embeds(attr_pos_as), 2, -1)
        attr_pvs = F.normalize(self.literal_embeds[attr_pos_vs.to(torch.long), :], 2, -1)
        final_phs = F.normalize(self.ent_embeds(attr_pos_hs), 2, -1)
        name_phs = F.normalize(self.name_embeds[attr_pos_hs.to(torch.long), :], 2, -1)

        '''attr_phs = self.av_ent_embeds(attr_pos_hs)
        attr_pas = self.attr_embeds(attr_pos_as)
        attr_pvs = self.literal_embeds[attr_pos_vs.to(torch.long), :]
        final_phs = self.ent_embeds(attr_pos_hs)
        name_phs = self.name_embeds[attr_pos_hs.to(torch.long), :]'''

        pos_score = self.conv(1, attr_phs, attr_pas, attr_pvs, self.args.dim)
        pos_score = torch.log(1 + torch.exp(-pos_score))
        pos_score = torch.multiply(pos_score, attr_pos_ws)
        pos_loss = torch.sum(pos_score)
        # print("attribute cv")
        pos_score = self.conv(2, final_phs, attr_pas, attr_pvs, self.args.dim)
        pos_loss += torch.sum(torch.log(1 + torch.exp(-pos_score)))
        pos_loss += 0.5 * alignment_loss(final_phs, name_phs)
        return pos_loss

    def conv(self, times, attr_hs, attr_as, attr_vs, dim, feature_map_size=2, kernel_size=None, activation=torch.tanh, layer_num=2):
        if kernel_size is None:
            kernel_size = [2, 4]
        attr_as = attr_as.view(-1, 1, dim)
        attr_vs = attr_vs.view(-1, 1, dim)

        input_avs = torch.concat([attr_as, attr_vs], 1)
        input_shape = input_avs.shape
        input_layer = input_avs.view(-1, input_shape[1], input_shape[2], 1)
        _conv = input_layer.to(torch.float32).permute(0, 2, 1, 3)
        _conv = self.batch_norm(_conv).permute(0, 3, 2, 1)
        _conv = torch.tanh(self.conv1(_conv))
        _conv = torch.tanh(self.conv2(_conv)).permute(0, 2, 3, 1)
        _conv = F.normalize(_conv, 2, 2)
        _shape = _conv.shape
        _flat = _conv.contiguous().view(-1, _shape[1] * _shape[2] * _shape[3])
        dense = torch.tanh(self.dense(_flat))
        dense = F.normalize(dense, 2, -1)  # important!!
        score = -torch.norm((attr_hs - dense), 2, -1)
        return score

    # --- The followings are cross-kg identity inference --- #

    def define_cross_kg_entity_reference_relation_view_graph(self, data):
        ckge_rel_pos_hs = data['ckge_rel_pos_hs']
        ckge_rel_pos_rs = data['ckge_rel_pos_rs']
        ckge_rel_pos_ts = data['ckge_rel_pos_ts']
        ckge_rel_phs = F.normalize(self.rv_ent_embeds(ckge_rel_pos_hs), 2, -1)
        ckge_rel_prs = F.normalize(self.rel_embeds(ckge_rel_pos_rs), 2, -1)
        ckge_rel_pts = F.normalize(self.rv_ent_embeds(ckge_rel_pos_ts), 2, -1)
        '''ckge_rel_phs = self.rv_ent_embeds(ckge_rel_pos_hs)
        ckge_rel_prs = self.rel_embeds(ckge_rel_pos_rs)
        ckge_rel_pts = self.rv_ent_embeds(ckge_rel_pos_ts)'''
        pos = torch.norm(ckge_rel_phs + ckge_rel_prs - ckge_rel_pts, 2, -1)
        ckge_relation_loss = 2 * torch.sum(pos)
        return ckge_relation_loss

    def define_cross_kg_entity_reference_attribute_view_graph(self, data):
        ckge_attr_pos_hs = data['ckge_attr_pos_hs']
        ckge_attr_pos_as = data['ckge_attr_pos_as']
        ckge_attr_pos_vs = data['ckge_attr_pos_vs']

        ckge_attr_phs = F.normalize(self.av_ent_embeds(ckge_attr_pos_hs), 2, -1)
        ckge_attr_pas = F.normalize(self.attr_embeds(ckge_attr_pos_as), 2, -1)
        ckge_attr_pvs = F.normalize(self.literal_embeds[ckge_attr_pos_vs.to(torch.long), :], 2, -1)
        '''ckge_attr_phs = self.av_ent_embeds(ckge_attr_pos_hs)
        ckge_attr_pas = self.attr_embeds(ckge_attr_pos_as)
        ckge_attr_pvs = self.literal_embeds[ckge_attr_pos_vs.to(torch.long), :]'''
        pos_score = self.conv(3, ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs, self.args.dim)
        return 2 * torch.sum(torch.log(1 + torch.exp(-pos_score)))

    def define_cross_kg_relation_reference_graph(self, data):
        ckgp_rel_pos_hs = data['ckgp_rel_pos_hs']
        ckgp_rel_pos_rs = data['ckgp_rel_pos_rs']
        ckgp_rel_pos_ts = data['ckgp_rel_pos_ts']
        ckgp_rel_pos_ws = data['ckgp_rel_pos_ws']
        ckgp_rel_phs = F.normalize(self.rv_ent_embeds(ckgp_rel_pos_hs), 2, -1)
        ckgp_rel_prs = F.normalize(self.rel_embeds(ckgp_rel_pos_rs), 2, -1)
        ckgp_rel_pts = F.normalize(self.rv_ent_embeds(ckgp_rel_pos_ts), 2, -1)
        '''ckgp_rel_phs = self.rv_ent_embeds(ckgp_rel_pos_hs)
        ckgp_rel_prs = self.rel_embeds(ckgp_rel_pos_rs)
        ckgp_rel_pts = self.rv_ent_embeds(ckgp_rel_pos_ts)'''
        return 2 * positive_loss_with_weight(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts,
                                                           ckgp_rel_pos_ws)

    def define_cross_kg_attribute_reference_graph(self, data):
        ckga_attr_pos_hs = data['ckga_attr_pos_hs']
        ckga_attr_pos_as = data['ckga_attr_pos_as']
        ckga_attr_pos_vs = data['ckga_attr_pos_vs']
        ckga_attr_pos_ws = data['ckga_attr_pos_ws']
        ckga_attr_phs = F.normalize(self.av_ent_embeds(ckga_attr_pos_hs), 2, -1)
        ckga_attr_pas = F.normalize(self.attr_embeds(ckga_attr_pos_as), 2, -1)
        ckga_attr_pvs = F.normalize(self.literal_embeds[ckga_attr_pos_vs.to(torch.long), :], 2, -1)
        '''ckga_attr_phs = self.av_ent_embeds(ckga_attr_pos_hs)
        ckga_attr_pas = self.attr_embeds(ckga_attr_pos_as)
        ckga_attr_pvs = self.literal_embeds[ckga_attr_pos_vs.to(torch.long), :]'''
        pos_score = self.conv(4, ckga_attr_phs, ckga_attr_pas, ckga_attr_pvs, self.args.dim)
        pos_score = torch.log(1 + torch.exp(-pos_score))
        pos_score = torch.multiply(pos_score, ckga_attr_pos_ws)
        pos_loss = torch.sum(pos_score)
        return pos_loss

    # --- The followings are intermediate combination --- #
    def define_common_space_learning_graph(self, data):
        cn_hs = data['cn_hs']

        final_cn_phs = F.normalize(self.ent_embeds(cn_hs), 2, -1)
        cn_hs_names = F.normalize(self.name_embeds[cn_hs.to(torch.long), :], 2, -1)
        cr_hs = F.normalize(self.rv_ent_embeds(cn_hs), 2, -1)
        ca_hs = F.normalize(self.av_ent_embeds(cn_hs), 2, -1)

        '''final_cn_phs = self.ent_embeds(cn_hs)
        cn_hs_names = self.name_embeds[cn_hs.to(torch.long), :]
        cr_hs = self.rv_ent_embeds(cn_hs)
        ca_hs = self.av_ent_embeds(cn_hs)'''

        cross_name_loss = alignment_loss(final_cn_phs, cn_hs_names)
        cross_name_loss += alignment_loss(final_cn_phs, cr_hs)
        cross_name_loss += alignment_loss(final_cn_phs, ca_hs)
        return cross_name_loss

    def define_space_mapping_graph(self, data):
        entities = data['entities']

        final_ents = F.normalize(self.ent_embeds(entities), 2, -1)
        nv_ents = F.normalize(torch.index_select(self.name_embeds, 0, entities), 2, -1)
        rv_ents = F.normalize(self.rv_ent_embeds(entities), 2, -1)
        av_ents = F.normalize(self.av_ent_embeds(entities), 2, -1)

        '''final_ents = self.ent_embeds(entities)
        nv_ents = self.name_embeds[entities.to(torch.long), :]
        rv_ents = self.rv_ent_embeds(entities)
        av_ents = self.av_ent_embeds(entities)'''

        nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, self.nv_mapping, self.eye_mat,
                                                   self.args.orthogonal_weight)
        rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, self.rv_mapping, self.eye_mat,
                                                   self.args.orthogonal_weight)
        av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, self.av_mapping, self.eye_mat,
                                                   self.args.orthogonal_weight)
        return nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss

    def eval_kg1_useful_ent_embeddings(self):
        embeds = F.normalize(self.ent_embeds(self.kgs.useful_entities_list1), 2, -1)
        #embeds = self.ent_embeds(self.kgs.useful_entities_list1)
        return embeds.detach().numpy()

    def eval_kg2_useful_ent_embeddings(self):
        embeds = F.normalize(self.ent_embeds(self.kgs.useful_entities_list2), 2, -1)
        #embeds = self.ent_embeds(self.kgs.useful_entities_list2)
        return embeds.detach().numpy()

    def test(self, entities1, entities2):
        seed_entity1 = F.normalize(self.ent_embeds(to_tensor(entities1)), 2, -1)
        seed_entity2 = F.normalize(self.ent_embeds(to_tensor(entities2)), 2, -1)
        _, _, _, sim_list = test(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), None,
                                 self.args.top_k, self.args.test_threads_num, metric=self.args.eval_metric, normalize=self.args.eval_norm,
                                 csls_k=0, accurate=True)
        print()
        return sim_list

    def valid(self, stop_metric):
        if len(self.kgs.valid_links) > 0:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.valid_entities2)), 2, -1)
        else:
            seed_entity1 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities1)), 2, -1)
            seed_entity2 = F.normalize(self.ent_embeds(to_tensor(self.kgs.test_entities2)), 2, -1)
        hits1_12, mrr_12 = valid(seed_entity1.detach().numpy(), seed_entity2.detach().numpy(), None,
                                 self.args.top_k,self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12



