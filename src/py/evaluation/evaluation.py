import gc
import os
import time
from math import cos, pi

import numpy as np
from tqdm import tqdm
from src.py.evaluation.alignment import greedy_alignment
from src.py.evaluation.similarity import sim
from src.py.util.util import task_divide


def get_score(h, r, t):
    a = h + r
    b = t
    '''h_norm = np.linalg.norm(x=h, ord=2, axis=-1, keepdims=True)
    r_norm = np.linalg.norm(x=r, ord=2, axis=-1, keepdims=True)
    t_norm = np.linalg.norm(x=t, ord=2, axis=-1, keepdims=True)
    h = h / h_norm
    r = r / r_norm
    t = t / t_norm
    score = (h + r) - t
    score = np.linalg.norm(score, 2, -1)'''
    tmp = min(a - b, 1 - (a - b))
    tmp = 2 * (1 - cos(2 * pi * tmp))
    return np.sum(tmp, -1) / 4


def get_rank(data, true, low_values=False):
    """Computes the rank of entity at index true[i]. If the rank is k then
    there are k-1 entities with better (higher or lower) value in data.

    Parameters
    ----------
    data: `torch.Tensor`, dtype: `torch.float`, shape: (n_facts, dimensions)
        Scores for each entity.
    true: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
        true[i] is the index of the true entity for test i of the batch.
    low_values: bool, optional (default=False)
        if True, best rank is the lowest score else it is the highest.

    Returns
    -------
    ranks: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
        ranks[i] - 1 is the number of entities which have better (or same)
        scores in data than the one and index true[i]
    """
    true_data = data[list(range(len(true))), true]
    true_data = np.expand_dims(true_data, 1)
    if low_values:
        return np.sum(data <= true_data, -1)
    else:
        return np.sum(data >= true_data, -1)


def get_true_targets(dictionary, key1, key2, true_idx, i):
    """For a current index `i` of the batch, returns a tensor containing the
    indices of entities for which the triplet is an existing one (i.e. a true
    one under CWA).

    Parameters
    ----------
    dictionary: default dict
        Dictionary of keys (int, int) and values list of ints giving all
        possible entities for the (entity, relation) pair.
    key1: torch.Tensor, shape: (batch_size), dtype: torch.long
    key2: torch.Tensor, shape: (batch_size), dtype: torch.long
    true_idx: torch.Tensor, shape: (batch_size), dtype: torch.long
        Tensor containing the true entity for each sample.
    i: int
        Indicates which index of the batch is currently treated.

    Returns
    -------
    true_targets: torch.Tensor, shape: (batch_size), dtype: torch.long
        Tensor containing the indices of entities such that
        (e_idx[i], r_idx[i], true_target[any]) is a true fact.

    """
    try:
        true_targets = dictionary[(key1[i], key2[i])].copy()
        if true_idx is not None:
            # true_targets.remove(true_idx[i])
            if len(true_targets) > 0:
                return list(true_targets)
            else:
                return None
        else:
            return list(true_targets)
    except KeyError:
        return None


def filter_scores(scores, dictionary, key1, key2, true_idx, low_value=True):
    # filter out the true negative samples by assigning - inf score.
    b_size = scores.shape[0]
    filt_scores = scores

    for i in range(b_size):
        true_targets = get_true_targets(dictionary, key1, key2, true_idx, i)
        if true_targets is None:
            continue
        if low_value:
            filt_scores[i][true_targets] = float('Inf')
        else:
            filt_scores[i][true_targets] = float('Inf')

    return filt_scores


def parse_triples(relation_set):
    subjects, predicates, objects = list(), list(), list()
    for o, s in relation_set:
        subjects.append(s)
        predicates.append(0)
        objects.append(o)
    return objects, predicates, subjects


class EntityTypeEvaluator:
    def __init__(self, model, args, kg, is_valid=False):
        self.evaluated = None
        self.rank_true_heads = np.array([])
        self.rank_true_tails = np.array([])
        self.filt_rank_true_heads = np.array([])
        self.filt_rank_true_tails = np.array([])
        self.model = model
        self.is_valid = is_valid
        self.args = args
        self.kg = kg
        # self.freeze_model()

    def fomulate(self, score):
        if self.args.is_torch:
            return score.cpu().detach().numpy()
        else:
            ''' needs to be implemented with tf'''
            return score

    def freeze_model(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def activate_model(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

    def trans_load_data(self):
        if self.is_valid:
            n = int(len(self.kg.valid_et_list) / 1000)
            kg1_list = task_divide(self.kg.valid_et_list, n)
        else:
            n = int(len(self.kg.test_et_list) / 1000)
            kg1_list = task_divide(self.kg.test_et_list, n)
        return kg1_list

    def evaluate(self):
        kg1_list = self.trans_load_data()
        # candidate = ent_embeds
        # candidate = np.expand_dims(candidate, 0).repeat(1000, axis=0)
        self.rank_true_heads = np.array([])
        self.rank_true_tails = np.array([])
        self.filt_rank_true_heads = np.array([])
        self.filt_rank_true_tails = np.array([])
        for triple_batch in tqdm(kg1_list):
            head, relation, tail = parse_triples(triple_batch)
            start = time.time()
            h_embeds, r_embeds, t_embeds, candidates = self.model.get_embeddings(head, relation, tail)
            score = self.fomulate(self.model.get_score(h_embeds, r_embeds, candidates))
            #filtered_score = filter_scores(score.copy(), self.kg.t_dict, head, relation, tail)
            filtered_score = score
            self.rank_true_tails = np.append(self.rank_true_tails, get_rank(score, tail, True))
            self.filt_rank_true_tails = np.append(self.filt_rank_true_tails, get_rank(filtered_score, tail, True))
            del h_embeds, r_embeds, t_embeds, candidates
            gc.collect()
        self.evaluated = True
        self.model.projected = False

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call ''LinkPredictionEvaluator.evaluate')
        sum_ = (np.mean(self.rank_true_tails))
        filt_sum = np.mean(self.filt_rank_true_tails)
        return sum_, filt_sum

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        tail_hit = np.mean(self.rank_true_tails <= k)
        filt_tail_hit = np.mean(self.filt_rank_true_tails <= k)

        return tail_hit, filt_tail_hit

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return tail_hit, filt_tail_hit

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        tail_mrr = np.mean(self.rank_true_tails ** (-1))
        filt_tail_mrr = np.mean(self.filt_rank_true_tails ** (-1))

        return tail_mrr, filt_tail_mrr

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        self.evaluate()
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))
        # self.activate_model()


class LinkPredictionEvaluator:
    def __init__(self, model, args, kg, is_valid=False):
        self.evaluated = None
        self.rank_true_heads = np.array([])
        self.rank_true_tails = np.array([])
        self.filt_rank_true_heads = np.array([])
        self.filt_rank_true_tails = np.array([])
        self.model = model
        self.is_valid = is_valid
        self.args = args
        self.kg = kg
        # self.freeze_model()

    def fomulate(self, score):
        if self.args.is_torch:
            return score.cpu().detach().numpy()
        else:
            ''' needs to be implemented with tf'''
            return score

    def freeze_model(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def activate_model(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

    def trans_load_data(self):
        if self.is_valid:
            n = int(len(self.kg.valid_relation_triples_list) / 1000)
            kg1_list = task_divide(self.kg.valid_relation_triples_list, n)
        else:
            n = int(len(self.kg.test_relation_triples_list) / 1000)
            kg1_list = task_divide(self.kg.test_relation_triples_list, n)
        return kg1_list

    def evaluate(self):
        kg1_list = self.trans_load_data()
        # candidate = ent_embeds
        # candidate = np.expand_dims(candidate, 0).repeat(1000, axis=0)
        self.rank_true_heads = np.array([])
        self.rank_true_tails = np.array([])
        self.filt_rank_true_heads = np.array([])
        self.filt_rank_true_tails = np.array([])
        for triple_batch in tqdm(kg1_list):
            head, relation, tail = parse_triples(triple_batch)
            start = time.time()
            h_embeds, r_embeds, t_embeds, candidates = self.model.get_embeddings(head, relation, tail)
            score = self.fomulate(self.model.get_score(h_embeds, r_embeds, candidates))
            filtered_score = filter_scores(score.copy(), self.kg.t_dict, head, relation, tail)

            self.rank_true_tails = np.append(self.rank_true_tails, get_rank(score, tail, True))
            self.filt_rank_true_tails = np.append(self.filt_rank_true_tails, get_rank(filtered_score, tail, True))
            score = self.fomulate(self.model.get_score(candidates, r_embeds, t_embeds))
            filtered_score = filter_scores(score.copy(), self.kg.h_dict, relation, tail, head)
            self.rank_true_heads = np.append(self.rank_true_heads, get_rank(score, head, True))
            self.filt_rank_true_heads = np.append(self.filt_rank_true_heads, get_rank(filtered_score, head, True))
            del h_embeds, r_embeds, t_embeds, candidates
            gc.collect()
        self.evaluated = True
        self.model.projected = False

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call ''LinkPredictionEvaluator.evaluate')
        sum_ = (np.mean(self.rank_true_heads) +
                np.mean(self.rank_true_tails))
        filt_sum = np.mean(self.filt_rank_true_heads) + np.mean(self.filt_rank_true_tails)
        return sum_ / 2, filt_sum / 2

    def hit_at_k_heads(self, k=10):
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        head_hit = np.mean(self.rank_true_heads <= k)
        filt_head_hit = np.mean(self.filt_rank_true_heads <= k)

        return head_hit, filt_head_hit

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        tail_hit = np.mean(self.rank_true_tails <= k)
        filt_tail_hit = np.mean(self.filt_rank_true_tails <= k)

        return tail_hit, filt_tail_hit

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')

        head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        head_mrr = np.mean(self.rank_true_heads ** (-1))
        tail_mrr = np.mean(self.rank_true_tails ** (-1))
        filt_head_mrr = np.mean(self.filt_rank_true_heads ** (-1))
        filt_tail_mrr = np.mean(self.filt_rank_true_tails ** (-1))

        return ((head_mrr + tail_mrr) / 2,
                (filt_head_mrr + filt_tail_mrr) / 2)

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        self.evaluate()
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))
        # self.activate_model()


class RelationPredictionEvaluator:
    def __init__(self, model, args, kg):
        self.evaluated = None
        self.rank_true_relations = np.array([])
        self.filt_rank_true_relations = np.array([])
        self.model = model
        self.args = args
        self.kg = kg

    def evaluate_test(self):
        class_name = self.model.__class__.__name__
        if class_name.find('Trans') > -1:
            self.trans_eval()

    def fomulate(self, score):
        if self.args.is_torch:
            return score.detach().numpy()
        else:
            ''' needs to be implemented with tf'''
            return score

    def trans_eval(self):
        dir = self.args.out_folder.split("/")
        new_dir = ""
        for i in range(len(dir) - 2):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + exist_file[0] + "/"
        ent_embeds = np.load(new_dir + "ent_embeds.npy")
        rel_embeds = np.load(new_dir + "rel_embeds.npy")
        kg1_list = task_divide(self.kg.test_relation_triples_list, self.args.batch_size)
        for i, triple_batch in kg1_list:
            head, relation, tail = parse_triples(triple_batch)
            batch_size = len(head)
            head_emb = np.expand_dims(ent_embeds[head], 1)
            tail_emb = np.expand_dims(ent_embeds[tail], 1)
            candidate = list(range(self.kg.relations_num))
            candidate = np.expand_dims(rel_embeds[candidate], 0).repeat(batch_size, axis=0)
            score = self.fomulate(self.model.calc(head_emb, candidate, tail_emb))
            filtered_score = filter_scores(score, self.kg.r_dict, head, tail, relation)
            self.rank_true_relations = np.append(self.rank_true_relations, get_rank(score, relation, True))
            self.filt_rank_true_relations = np.append(self.filt_rank_true_relations,
                                                      get_rank(filtered_score, relation, True))
        self.evaluated = True

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call ''RealtionPredictionEvaluator.evaluate')
        sum_ = np.mean(self.rank_true_relations)
        filt_sum = np.mean(self.filt_rank_true_relations)
        return sum_ / 2, filt_sum / 2

    def hit_at_k_relations(self, k=10):
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'RelationPredictionEvaluator.evaluate')
        relation_hit = np.mean(self.rank_true_relations <= k)
        filt_relation_hit = np.mean(self.filt_rank_true_relations <= k)

        return relation_hit, filt_relation_hit

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'RelationPredictionEvaluator.evaluate')

        head_hit, filt_head_hit = self.hit_at_k_relations(k=k)

        return head_hit, filt_head_hit

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise Exception('Evaluator not evaluated call '
                            'LinkPredictionEvaluator.evaluate')
        relation_mrr = np.mean(self.rank_true_relations ** (-1))
        filt_relation_mrr = np.mean(self.filt_rank_true_relations ** (-1))

        return relation_mrr, filt_relation_mrr

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))


def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False):
    if mapping is None:
        _, hits1_12, mr_12, mrr_12, _ = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                         metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12, _ = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                         metric, normalize, csls_k, accurate)
    return hits1_12, mrr_12


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                                metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(test_embeds1_mapped, embeds2, top_k,
                                                                                threads_num, metric, normalize, csls_k,
                                                                                accurate)
    return alignment_rest_12, hits1_12, mrr_12, sim_list


class entity_alignment_evaluation:
    def __init__(self, model, args, kgs):
        self.model = model
        self.kgs = kgs
        self.args = args

    def test(self, top_k=1, min_sim_value=None, output_file_name=None):
        """
        Compute pairwise similarity between the two collections of embeddings.
        Parameters
        ----------
        top_k : int
            The k for top k retrieval, can be None (but then min_sim_value should be set).
        min_sim_value : float, optional
            the minimum value for the confidence.
        output_file_name : str, optional
            The name of the output file. It is formatted as tsv file with entity1, entity2, confidence.
        Returns
        -------
        topk_neighbors_w_sim : A list of tuples of form (entity1, entity2, confidence)
        """
        embeds1 = self.model.ent_npy[self.kgs.test_entities1]
        embeds2 = self.model.ent_npy[self.kgs.test_entities2]
        mapping = None

        print(self.__class__.__name__, type(self.__class__.__name__))
        if self.__class__.__name__ == "GCN_Align":
            attr_embeds1 = self.attr_npy[self.kgs.test_entities1]
            attr_embeds2 = self.attr_npy[self.kgs.test_entities2]
            embeds1 = np.concatenate([embeds1 * self.args.beta, attr_embeds1 * (1.0 - self.args.beta)], axis=1)
            embeds2 = np.concatenate([embeds2 * self.args.beta, attr_embeds2 * (1.0 - self.args.beta)], axis=1)

        # if self.__class__.__name__ == "MTransE" or self.__class__.__name__ == "SEA" or self.__class__.__name__ == "KDCoE":
        if self.model.map_npy is not None:
            mapping = self.model.map_npy

        # search for correspondences which match top_k and/or min_sim_value
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)

        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)

    def predict_entities(self, entities_file_path, output_file_name=None):
        """
        Compute the confidence of given entities if they match or not.
        Parameters
        ----------
        entities_file_path : str
            A path pointing to a file formatted as (entity1, entity2) with tab separated (tsv-file).
            If given, the similarity of the entities is retrieved and returned (or also written to file if output_file_name is given).
            The parameters top_k and min_sim_value do not play a role, if this parameter is set.
        output_file_name : str, optional
            The name of the output file. It is formatted as tsv file with entity1, entity2, confidence.
        Returns
        -------
        topk_neighbors_w_sim : A list of tuples of form (entity1, entity2, confidence)
        """

        kg1_entities = list()
        kg2_entities = list()
        with open(entities_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                entities = line.strip('\n').split('\t')
                kg1_entities.append(self.kgs.kg1.entities_id_dict[entities[0]])
                kg2_entities.append(self.kgs.kg2.entities_id_dict[entities[1]])
        kg1_distinct_entities = list(set(kg1_entities))  # make distinct
        kg2_distinct_entities = list(set(kg2_entities))

        kg1_mapping = {entity_id: index for index, entity_id in enumerate(kg1_distinct_entities)}
        kg2_mapping = {entity_id: index for index, entity_id in enumerate(kg2_distinct_entities)}

        embeds1 = self.model.ent_embeds(kg1_distinct_entities).detach().numpy()
        embeds2 = self.model.ent_embeds(kg2_distinct_entities).detach().numpy()

        if self.model.mapping_mat:
            embeds1 = np.matmul(embeds1, self.model.mapping_mat.weight.data)

        sim_mat = sim(embeds1, embeds2, metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0)

        # map back with entities_id_dict to be sure that the right uri is chosen
        kg1_id_to_uri = {v: k for k, v in self.kgs.kg1.entities_id_dict.items()}
        kg2_id_to_uri = {v: k for k, v in self.kgs.kg2.entities_id_dict.items()}

        topk_neighbors_w_sim = []
        for entity1_id, entity2_id in zip(kg1_entities, kg2_entities):
            topk_neighbors_w_sim.append((
                kg1_id_to_uri[entity1_id],
                kg2_id_to_uri[entity2_id],
                sim_mat[kg1_mapping[entity1_id], kg2_mapping[entity2_id]]
            ))

        if output_file_name is not None:
            # create dir if not existent
            if not os.path.exists(self.model.out_folder):
                os.makedirs(self.model.out_folder)
            with open(self.model.out_folder + output_file_name, 'w', encoding='utf8') as file:
                for entity1, entity2, confidence in topk_neighbors_w_sim:
                    file.write(str(entity1) + "\t" + str(entity2) + "\t" + str(confidence) + "\n")
            print(self.model.out_folder + output_file_name, "saved")

        return topk_neighbors_w_sim

