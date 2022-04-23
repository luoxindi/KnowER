import gc
import math
import random
import time

import numpy as np
from joblib._multiprocessing_helpers import mp
from torch.autograd import Variable
from src.approaches.mtranse import MTransE
import torch
import torch.nn as nn
import src.trainer.batch as bat
from src.approaches.multike import MultiKE, generate_attribute_triple_batch_queue, valid_temp
from src.modules.data_loader import LoadDataset
from src.modules.finding.alignment import task_divide
from src.trainer.util import get_optimizer, to_tensor


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False


def to_var(batch):
    return Variable(torch.from_numpy(np.array(batch)))


class multike_trainer:
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = None
        self.model = MultiKE(args, kgs)
        self.model.init()
        self.optimizer = get_optimizer(self.args.optimizer, self.model, self.args.learning_rate)

    def train_relation_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.local_relation_triples_list, self.kgs.kg2.local_relation_triples_list,
                             self.kgs.kg1.local_relation_triples_set, self.kgs.kg2.local_relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            self.optimizer.zero_grad()
            batch_loss = self.model.define_relation_view_graph({'rel_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                                    'rel_pos_rs': to_tensor([x[1] for x in batch_pos]),
                                                                    'rel_pos_ts': to_tensor([x[2] for x in batch_pos]),
                                                                    'rel_neg_hs': to_tensor([x[0] for x in batch_neg]),
                                                                    'rel_neg_rs': to_tensor([x[1] for x in batch_neg]),
                                                                    'rel_neg_ts': to_tensor([x[2] for x in batch_neg])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.local_relation_triples_list)
        random.shuffle(self.kgs.kg2.local_relation_triples_list)
        end = time.time()
        print('epoch {} of rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    def train_attribute_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=generate_attribute_triple_batch_queue,
                       args=(self.model.predicate_align_model.attribute_triples_w_weights1,
                             self.model.predicate_align_model.attribute_triples_w_weights2,
                             self.model.predicate_align_model.attribute_triples_w_weights_set1,
                             self.model.predicate_align_model.attribute_triples_w_weights_set2,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.attribute_batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, 0)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            self.optimizer.zero_grad()
            batch_loss = self.model.define_attribute_view_graph({'attr_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                                 'attr_pos_as': to_tensor([x[1] for x in batch_pos]),
                                                                 'attr_pos_vs': to_tensor([x[2] for x in batch_pos]),
                                                                 'attr_pos_ws': to_tensor([x[3] for x in batch_pos])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.model.predicate_align_model.attribute_triples_w_weights1)
        random.shuffle(self.model.predicate_align_model.attribute_triples_w_weights2)
        end = time.time()
        print('epoch {} of att. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    # --- The followings are training for cross-kg identity inference --- #

    def train_cross_kg_entity_inference_relation_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_cross_kg_entity_reference_relation_view_graph({'ckge_rel_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                        'ckge_rel_pos_rs': to_tensor([x[1] for x in batch_pos]),
                                                        'ckge_rel_pos_ts': to_tensor([x[2] for x in batch_pos])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                            epoch_loss,
                                                                                                            end - start))

    def train_cross_kg_entity_inference_attribute_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_cross_kg_entity_reference_attribute_view_graph({'ckge_attr_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                        'ckge_attr_pos_as': to_tensor([x[1] for x in batch_pos]),
                                                        'ckge_attr_pos_vs': to_tensor([x[2] for x in batch_pos])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                             epoch_loss,
                                                                                                             end - start))

    def train_cross_kg_relation_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_cross_kg_relation_reference_graph({'ckgp_rel_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                        'ckgp_rel_pos_rs': to_tensor([x[1] for x in batch_pos]),
                                                        'ckgp_rel_pos_ts': to_tensor([x[2] for x in batch_pos]),
                                                        'ckgp_rel_pos_ws': to_tensor([x[3] for x in batch_pos])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg relation inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                              epoch_loss,
                                                                                                              end - start))

    def train_cross_kg_attribute_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_cross_kg_attribute_reference_graph({'ckga_attr_pos_hs': to_tensor([x[0] for x in batch_pos]),
                                                        'ckga_attr_pos_as': to_tensor([x[1] for x in batch_pos]),
                                                        'ckga_attr_pos_vs': to_tensor([x[2] for x in batch_pos]),
                                                        'ckga_attr_pos_ws': to_tensor([x[3] for x in batch_pos])})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg attribute inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'
              .format(epoch, epoch_loss, end - start))

    def train_shared_space_mapping_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_space_mapping_graph({'entities': to_tensor(batch_pos)})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of shared space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    # --- The followings are training for cross-view inference --- #

    def train_common_space_learning_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            self.optimizer.zero_grad()
            batch_loss = self.model.define_common_space_learning_graph({'cn_hs': to_tensor(batch_pos)})
            batch_loss.backward()
            self.optimizer.step()
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of common space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    def test(self):
        self.model.test(self.kgs.test_entities1, self.kgs.test_entities2)

    def run(self):
        t = time.time()
        relation_triples_num = self.kgs.kg1.local_relation_triples_num + self.kgs.kg2.local_relation_triples_num
        attribute_triples_num = self.kgs.kg1.local_attribute_triples_num + self.kgs.kg2.local_attribute_triples_num
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        attribute_batch_queue = manager.Queue()
        cross_kg_relation_triples = self.kgs.kg1.sup_relation_triples_list + self.kgs.kg2.sup_relation_triples_list
        cross_kg_entity_inference_in_attribute_triples = self.kgs.kg1.sup_attribute_triples_list + \
                                                         self.kgs.kg2.sup_attribute_triples_list
        cross_kg_relation_inference = self.model.predicate_align_model.sup_relation_alignment_triples1 + \
                                      self.model.predicate_align_model.sup_relation_alignment_triples2
        cross_kg_attribute_inference = self.model.predicate_align_model.sup_attribute_alignment_triples1 + \
                                       self.model.predicate_align_model.sup_attribute_alignment_triples2
        neighbors1, neighbors2 = None, None
        entity_list = self.kgs.kg1.entities_list + self.kgs.kg2.entities_list

        valid_temp(self.model, embed_choice='nv')
        for i in range(1, self.args.max_epoch + 1):
            print('epoch {}:'.format(i))
            self.train_relation_view_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue,
                                          neighbors1, neighbors2)
            self.train_common_space_learning_1epo(i, entity_list)
            self.train_cross_kg_entity_inference_relation_view_1epo(i, cross_kg_relation_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_relation_inference_1epo(i, cross_kg_relation_inference)

            self.train_attribute_view_1epo(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue,
                                           neighbors1, neighbors2)
            self.train_common_space_learning_1epo(i, entity_list)
            self.train_cross_kg_entity_inference_attribute_view_1epo(i, cross_kg_entity_inference_in_attribute_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_attribute_inference_1epo(i, cross_kg_attribute_inference)

            if i >= 10 and i % self.args.eval_freq == 0:
                valid_temp(self.model, embed_choice='rv')
                valid_temp(self.model, embed_choice='av')
                # valid_temp(self, embed_choice='final')
                # valid_temp(self, embed_choice='avg')
                # valid_WVA(self)
                flag = self.model.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break

            if i >= self.args.start_predicate_soft_alignment and i % 10 == 0:
                self.model.predicate_align_model.update_predicate_alignment(self.model.rel_embeds.weight.data)
                self.model.predicate_align_model.update_predicate_alignment(self.model.attr_embeds.weight.data,
                                                                      predicate_type='attribute')
                cross_kg_relation_inference = self.model.predicate_align_model.sup_relation_alignment_triples1 + \
                                              self.model.predicate_align_model.sup_relation_alignment_triples2
                cross_kg_attribute_inference = self.model.predicate_align_model.sup_attribute_alignment_triples1 + \
                                               self.model.predicate_align_model.sup_attribute_alignment_triples2

            # if self.early_stop or i == self.args.max_epoch:
            #     break

            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours(self.model.eval_kg1_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list1,
                                                     neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.model.eval_kg2_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list2,
                                                     neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()

        # for i in range(1, self.args.shared_learning_max_epoch + 1):
        #     self.train_shared_space_mapping_1epo(i, entity_list)
        #     if i >= self.args.start_valid and i % self.args.eval_freq == 0:
        #         self.valid(self)
        # test_WVA(self)

        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
