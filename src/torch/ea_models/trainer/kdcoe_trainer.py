import gc
import math
import random
import string
import time
import pandas as pd
import numpy as np
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable

from src.approaches.kdcoe import KDCoE
from src.approaches.mtranse import MTransE
import torch
import torch.nn as nn

from src.models.basic_model import align_model_trainer
from src.modules.data_loader import LoadDataset
from src.modules.finding.alignment import task_divide, check_new_alignment, search_nearest_k, find_alignment
from src.modules.finding.evaluation import valid, test
from src.modules.finding.similarity import sim
from src.trainer.util import get_optimizer, to_tensor


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def to_var(batch):
    return Variable(torch.from_numpy(np.array(batch)))


class kdcoe_trainer(align_model_trainer):
    def __init__(self):
        super(kdcoe_trainer, self).__init__()
        self.ref_entities2 = None
        self.ref_entities1 = None
        self.e_desc = None
        self.word_em = None
        self.sim_th = None
        self.desc_sim_th = None
        self.word_embed = None
        self.default_desc_length = None
        self.wv_dim = None
        self.negative_indication_weight = None
        self.desc_batch_size = None
        self.new_alignment_index = set()
        self.new_alignment = set()
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.optimizer = None

    def init(self, args, kgs):
        self.desc_batch_size = self.args.desc_batch_size
        self.negative_indication_weight = -1. / self.desc_batch_size
        self.wv_dim = self.args.wv_dim
        self.default_desc_length = self.args.default_desc_length
        self.word_embed = self.args.word_embed
        self.desc_sim_th = self.args.desc_sim_th
        self.sim_th = self.args.sim_th
        self.word_em, self.e_desc = self._get_desc_input()
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.model = KDCoE(args, kgs)
        self.model.init()

    def _get_desc_input(self):

        list1 = self.kgs.train_entities1 + self.kgs.valid_entities1 + self.kgs.test_entities1
        list2 = self.kgs.train_entities2 + self.kgs.valid_entities2 + self.kgs.test_entities2
        aligned_dict = dict(zip(list1, list2))
        print("aligned dict", len(aligned_dict))

        # desc graph settings
        start = time.time()
        # find desc
        model = self
        at1 = pd.DataFrame(model.kgs.kg1.attribute_triples_list)
        at2 = pd.DataFrame(model.kgs.kg2.attribute_triples_list)
        """
               0    1                                                  2
        0  22816  168  "4000.1952"^^<http://www.w3.org/2001/XMLSchema...
        1  14200    6  "1.82"^^<http://www.w3.org/2001/XMLSchema#double>
        2  20874   38                                              99657
        """
        # 将属性和其对应的id进行反转
        aid1 = pd.Series(list(model.kgs.kg1.attributes_id_dict), index=model.kgs.kg1.attributes_id_dict.values())
        aid2 = pd.Series(list(model.kgs.kg2.attributes_id_dict), index=model.kgs.kg2.attributes_id_dict.values())
        """
        0                         http://xmlns.com/foaf/0.1/name
        2                  http://dbpedia.org/ontology/birthDate
        """
        """
        1                      http://dbpedia.org/ontology/years
        3        http://dbpedia.org/ontology/appearancesInLeague
        """
        uri_name = 'escription'  # in Wikidata, the attribute is http://schema.org/description
        desc_uris1 = aid1[aid1.str.findall(uri_name).apply(lambda x: len(x)) > 0]
        desc_uris2 = aid2[aid2.str.findall(uri_name).apply(lambda x: len(x)) > 0]
        """
        8      http://purl.org/dc/elements/1.1/description
        462        http://dbpedia.org/ontology/description
        464    http://dbpedia.org/ontology/depictionDescription
        """
        """
        31     http://dbpedia.org/ontology/depictionDescription
        123           http://purl.org/dc/terms/description
        183    http://purl.org/dc/elements/1.1/description
        """
        desc_ids1 = desc_uris1.index.values.tolist()
        desc_ids2 = desc_uris2.index.values.tolist()
        """
        [31 123 183]
        """
        e_desc1 = at1[at1.iloc[:, 1].isin(desc_ids1)]
        e_desc2 = at2[at2.iloc[:, 1].isin(desc_ids2)]
        print("kg1 descriptions:", len(e_desc1))
        print("kg2 descriptions:", len(e_desc2))
        """
        156083   7169   31                        Tomasz Wisio (2016, rechts)
        156127   1285   31     Olk (links) mitFranz BeckenbauerundGerd Müller
        """
        e_desc1 = e_desc1.drop_duplicates(subset=0)
        e_desc2 = e_desc2.drop_duplicates(subset=0)
        print("after drop_duplicates, kg1 descriptions:", len(e_desc1))
        print("after drop_duplicates, kg2 descriptions:", len(e_desc2))

        ents_w_desc1_list = e_desc1.iloc[:, 0].values.tolist()
        ents_w_desc1 = set(ents_w_desc1_list)
        ents_w_desc1_index = e_desc1.index.values.tolist()
        print("kg1 entities having descriptions:", len(ents_w_desc1))
        ents_w_desc2_list = e_desc2.iloc[:, 0].values.tolist()
        ents_w_desc2 = set(ents_w_desc2_list)
        print("kg2 entities having descriptions:", len(ents_w_desc2))

        # drop_desc_index1 = []
        # selected_ent2_ids = []
        # for i in range(len(ents_w_desc1_list)):
        #     aligned_ent2 = aligned_dict.get(ents_w_desc1_list[i], None)
        #     if aligned_ent2 not in ents_w_desc2:
        #         drop_desc_index1.append(ents_w_desc1_index[i])
        #     else:
        #         selected_ent2_ids.append(aligned_ent2)
        # e_desc1 = e_desc1.drop(drop_desc_index1)
        # e_desc2 = e_desc2[e_desc2.iloc[:, 0].isin(selected_ent2_ids)]
        # print("after alignment, kg1 descriptions:", len(e_desc1))
        # print("after alignment, kg2 descriptions:", len(e_desc2))
        # ents_w_desc1_list = e_desc1.iloc[:, 0].values.tolist()
        # ents_w_desc1 = set(ents_w_desc1_list)
        # ents_w_desc2_list = e_desc2.iloc[:, 0].values.tolist()
        # ents_w_desc2 = set(ents_w_desc2_list)
        # print("after alignment, kg1 entities having descriptions:", len(ents_w_desc1))
        # print("after alignment, kg2 entities having descriptions:", len(ents_w_desc2))

        # prepare desc
        e_desc1.iloc[:, 2] = e_desc1.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
        e_desc2.iloc[:, 2] = e_desc2.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
        """
        155791                            [Tiffeny, Milbrett, 2003]
        155801        [Plattspitzen, von, Nordosten, Jubiläumsgrat]
        """
        name_triples = self._get_local_name_by_name_triple()
        names = pd.DataFrame(name_triples)
        names.iloc[:, 2] = names.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
        names.iloc[e_desc1.iloc[:, 0].values, [1, 2]] = e_desc1.iloc[:, [1, 2]].values
        names.iloc[e_desc2.iloc[:, 0].values, [1, 2]] = e_desc2.iloc[:, [1, 2]].values
        """
        29998  29998 -1                                      [Til, Death]
        29999  29999 -1  [You, Gotta, Fight, for, Your, Right, to, Party]
        """

        # load word embedding
        with open(self.word_embed, 'r', encoding='utf-8') as f:
            w = f.readlines()
            w = pd.Series(w[1:])

        we = w.str.split(' ')
        word = we.apply(lambda x: x[0])
        w_em = we.apply(lambda x: x[1:])
        del we
        gc.collect()
        print('concat word embeddings')
        word_em = np.stack(w_em.values, axis=0).astype(np.float)
        word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
        print('convert words to ids')
        w_in_desc = []
        for l in names.iloc[:, 2].values:
            w_in_desc += l
        w_in_desc = pd.Series(list(set(w_in_desc)))
        un_logged_words = w_in_desc[~w_in_desc.isin(word)]
        un_logged_id = len(word)

        all_word = pd.concat(
            [pd.Series(word.index, word.values),
             pd.Series([un_logged_id, ] * len(un_logged_words), index=un_logged_words)])

        def lookup_and_padding(x):
            default_length = 4
            ids = list(all_word.loc[x].values) + [all_word.iloc[-1], ] * default_length
            return ids[:default_length]

        print('look up desc embeddings')
        names.iloc[:, 2] = names.iloc[:, 2].apply(lookup_and_padding)

        # entity-desc-embedding dataframe
        e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], model.kgs.entities_num, axis=0),
                                    range(model.kgs.entities_num))

        e_desc_input.iloc[names.iloc[:, 0].values] = np.stack(names.iloc[:, 2].values)

        print('generating desc input costs time: {:.4f}s'.format(time.time() - start))
        return word_em, e_desc_input

    def _get_local_name_by_name_triple(self, name_attribute_list=None):
        if name_attribute_list is None:
            if 'D_Y' in self.args.training_data:
                name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
            elif 'D_W' in self.args.training_data:
                name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
            else:
                name_attribute_list = {}

        local_triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        triples = list()
        for h, a, v in local_triples:
            v = v.strip('"')
            if v.endswith('"@eng'):
                v = v.rstrip('"@eng')
            triples.append((h, a, v))
        id_ent_dict = {}
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e

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
        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        for (e, a, v) in triples:
            if a in name_ids:
                local_name_dict[e] = v
        for e in ents:
            if e not in local_name_dict:
                local_name_dict[e] = id_ent_dict[e].split('/')[-1].replace('_', ' ')
        name_triples = list()
        for e, n in local_name_dict.items():
            name_triples.append((e, -1, n))
        return name_triples

    def valid_desc(self, stop_metric):
        print("valid desc")
        valid_links = pd.DataFrame(self.kgs.valid_links)
        desc1 = self.e_desc.loc[valid_links.values[:, 0]].values
        desc2 = self.e_desc.loc[valid_links.values[:, 1]].values
        desc_em1 = self.word_em[desc1]
        desc_em2 = self.word_em[desc2]
        loss, dem1, dem2 = self.model.define_desc_graph({
            'desc1': to_tensor(desc_em1),
            'desc2': to_tensor(desc_em2)
        })
        hits1_12, mrr_12 = valid(dem1, dem2, None, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        # del dem1, dem2, desc_em1, desc_em2, desc1, desc2
        # gc.collect()
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def test_desc(self):
        print("test desc")
        test_links = pd.DataFrame(self.kgs.test_links)
        desc1 = self.e_desc.loc[test_links.values[:, 0]].values
        desc2 = self.e_desc.loc[test_links.values[:, 1]].values
        desc_em1 = self.word_em[desc1]
        desc_em2 = self.word_em[desc2]
        dem1, dem2 = self.model.define_desc_graph({
            'desc1': to_tensor(desc_em1),
            'desc2': to_tensor(desc_em2)
        })
        test(dem1, dem2, None, self.args.top_k,
             self.args.test_threads_num, metric=self.args.eval_metric,
             normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_desc_1epo(self, epoch):
        start = time.time()

        if len(self.new_alignment) > 0:
            alignments = pd.DataFrame(self.kgs.train_links + list(self.new_alignment))
        else:
            alignments = pd.DataFrame(self.kgs.train_links)

        word_em = self.word_em
        e_desc = self.e_desc

        batch_size = self.desc_batch_size
        if batch_size > len(alignments):
            batch_size = len(alignments)
        batch_num = len(alignments) // batch_size

        # shuffle
        choices = np.random.choice(len(alignments), size=len(alignments), replace=True)
        epoch_loss = 0.0
        for i in range(batch_num):
            one_batch_choice = choices[i * batch_size:(i + 1) * batch_size]
            one_batch_alignments = alignments.iloc[one_batch_choice]

            desc1 = e_desc.loc[one_batch_alignments.values[:, 0]].values
            desc2 = e_desc.loc[one_batch_alignments.values[:, 1]].values

            desc_em1 = word_em[desc1]
            desc_em2 = word_em[desc2]
            self.optimizer.zero_grad()
            desc_loss = self.model.define_desc_graph({
                'desc1': to_tensor(desc_em1),
                'desc2': to_tensor(desc_em2)
            })
            desc_loss.backward()
            self.optimizer.step()
            epoch_loss += desc_loss
        print('epoch {}, avg. desc loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_triple_training_1epo(self, epoch, triple_steps, data_loader_kg1, data_loader_kg2, neighbors1,
                                    neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            data1 = data_loader_kg1.__next__()
            data2 = data_loader_kg2.__next__()
            trained_samples_num += (len(data1['pos_hs']) + len(data2['pos_hs']))
            batch_loss = self.model.generate_transE_loss(
                {'pos_hs': to_var(np.concatenate((data1['pos_hs'], data2['pos_hs']))),
                 'pos_rs': to_var(np.concatenate((data1['pos_rs'], data2['pos_rs']))),
                 'pos_ts': to_var(np.concatenate((data1['pos_ts'], data2['pos_ts']))),
                 'neg_hs': to_var(np.concatenate((data1['neg_hs'], data2['neg_hs']))),
                 'neg_rs': to_var(np.concatenate((data1['neg_rs'], data2['neg_rs']))),
                 'neg_ts': to_var(np.concatenate((data1['neg_ts'], data2['neg_ts'])))})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            self.optimizer.zero_grad()
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            batch_loss = self.model.generate_mapping_loss({'seed1': to_var(np.array([x[0] for x in links_batch])),
                                                           'seed2': to_var(np.array([x[1] for x in links_batch]))})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_mapping_training_1epo_new(self, epoch, triple_steps):
        t = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        if self.new_alignment is not None and len(self.new_alignment) > 0:
            train_links = list(self.new_alignment)
        else:
            return
        batch_size = math.ceil(len(train_links) / triple_steps)
        for i in range(triple_steps):
            links_batch = random.sample(train_links, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.generate_mapping_loss({'seed1': to_var(np.array([x[0] for x in links_batch])),
                                                           'seed2': to_var(np.array([x[1] for x in links_batch]))})
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. new mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - t))

    def find_new_alignment_rel(self):
        t = time.time()
        un_aligned_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        un_aligned_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        sim_mat = self.model.find_new_alignment_rel(un_aligned_ent1,  un_aligned_ent2)
        print("find new alignment based on relational embeddings:")
        new_alignment_rel_index = find_alignment(sim_mat, self.sim_th, 1)
        check_new_alignment(new_alignment_rel_index)
        if new_alignment_rel_index is None or len(new_alignment_rel_index) == 0:
            return False
        stop = False
        if len(self.new_alignment_index) == 0:
            self.new_alignment_index = set(new_alignment_rel_index)
        elif len(set(new_alignment_rel_index) - self.new_alignment_index) == 0:
            stop = True
        else:
            self.new_alignment_index |= set(new_alignment_rel_index)
            stop = False
        check_new_alignment(self.new_alignment_index, context='check total new alignment')
        self.new_alignment = [(un_aligned_ent1[x], un_aligned_ent2[y]) for (x, y) in self.new_alignment_index]
        # del embeds1, embeds2, sim_mat
        print('finding new alignment costs time: {:.4f}s'.format(time.time() - t))
        return stop

    def find_new_alignment_desc(self):
        t = time.time()
        print("sim th", self.desc_sim_th, self.sim_th)
        # find new alignment based on description embeddings
        print("find new alignment based on description embeddings:")
        to_align_links = pd.DataFrame(self.kgs.valid_links + self.kgs.test_links)
        un_aligned_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        un_aligned_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        desc1 = self.e_desc.loc[to_align_links.values[:, 0]].values
        desc2 = self.e_desc.loc[to_align_links.values[:, 1]].values
        desc_em1 = self.word_em[desc1]
        desc_em2 = self.word_em[desc2]
        dem1, dem2 = self.model.define_desc_graph({
            'desc1': to_tensor(desc_em1),
            'desc2': to_tensor(desc_em2)
        })
        num = len(self.kgs.valid_links + self.kgs.test_links)
        search_tasks = task_divide(np.array(range(num)), self.args.test_threads_num // 2)

        pool = mp.Pool(processes=len(search_tasks))
        rests = list()
        for indexes in search_tasks:
            sub_embeds = dem1[indexes, :]
            rests.append(pool.apply_async(find_alignment, (sub_embeds, dem2, indexes, self.desc_sim_th)))
        pool.close()
        pool.join()
        new_alignment_desc_index = set()
        for rest in rests:
            alignment = rest.get()
            new_alignment_desc_index |= set(alignment)

        # new_alignment_desc_index = set()
        # for indexes in search_tasks:
        #     sub_embeds = dem1[indexes, :]
        #     alignment = find_alignment(sub_embeds, dem2, indexes, self.desc_sim_th)
        #     new_alignment_desc_index |= set(alignment)

        if new_alignment_desc_index is None or len(new_alignment_desc_index) == 0:
            return False
        check_new_alignment(new_alignment_desc_index)
        stop = False
        if len(self.new_alignment_index) == 0:
            self.new_alignment_index = set(new_alignment_desc_index)
        elif len(set(new_alignment_desc_index) - self.new_alignment_index) == 0:
            stop = True
        else:
            self.new_alignment_index |= set(new_alignment_desc_index)
            stop = False
        check_new_alignment(self.new_alignment_index, context='check total new alignment')
        self.new_alignment = [(un_aligned_ent1[x], un_aligned_ent2[y]) for (x, y) in self.new_alignment_index]
        # del desc_em1, desc_em2, dem1, dem2, desc_sim
        print('finding new alignment costs time: {:.4f}s'.format(time.time() - t))
        return stop

    def test(self):
        rest_12 = self.model.tests(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        kg1_size = int((self.kgs.kg1.relation_triples_num / triples_num) * self.args.batch_size)
        kg2_size = self.args.batch_size - kg1_size
        triple_times1 = int(self.kgs.kg1.relation_triples_num / kg1_size)
        triple_times2 = int(self.kgs.kg2.relation_triples_num / kg2_size)
        self.optimizer = get_optimizer(self.args.optimizer, self.model, self.args.learning_rate)
        data_loader1 = LoadDataset(self.kgs.kg1, kg1_size, self.args.batch_threads_num,
                                   self.args.neg_triple_num)
        data_loader2 = LoadDataset(self.kgs.kg2, kg2_size, self.args.batch_threads_num,
                                   self.args.neg_triple_num)
        train_steps = int(math.ceil(triples_num / self.args.batch_size))
        for it in range(1, self.args.max_iter + 1):
            self.flag1 = -1
            self.flag2 = -1
            data_loader_kg1 = iter(data_loader1)
            data_loader_kg2 = iter(data_loader2)
            self.early_stop = False
            for i in range(1, self.args.max_epoch + 1):
                self.launch_desc_1epo(i)
                if i > 0 and i % self.args.eval_freq == 0:
                    # gc.collect()
                    flag = self.valid_desc(self.args.stop_metric)
                    # gc.collect()
                    self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                    if self.early_stop or i == self.args.max_epoch:
                        break
            gc.collect()
            stop = self.find_new_alignment_desc()
            gc.collect()
            if stop:
                print("co-training ends")
                break
            self.flag1 = -1
            self.flag2 = -1
            self.early_stop = False
            for i in range(1, self.args.max_epoch + 1):
                self.launch_triple_training_1epo(i, train_steps, data_loader_kg1, data_loader_kg2, None, None)
                self.launch_mapping_training_1epo(i, train_steps)
                self.launch_mapping_training_1epo_new(i, train_steps)
                if i > 0 and i % self.args.eval_freq == 0:
                    flag = self.model.valid(self.args.stop_metric)
                    # gc.collect()
                    self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                    if self.early_stop or i == self.args.max_epoch:
                        break
            stop = self.find_new_alignment_rel()
            # gc.collect()
            if stop:
                print("co-training ends")
                break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
