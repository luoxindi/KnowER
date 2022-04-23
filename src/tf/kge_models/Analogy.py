import math

import tensorflow as tf
from .basic_model import BasicModel
from ...py.base.initializers import init_embeddings, init_embeddings_v2
from ...py.load import read
from ...py.util.util import to_var


class Analogy(BasicModel):

    def __init__(self, kgs, args, dim=100):
        super(Analogy, self).__init__(args, kgs)
        self.dim = dim
        self.ent_re_embeddings = init_embeddings_v2([self.ent_tot, self.dim], 'ent_re_embeddings',
                                          self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.ent_im_embeddings = init_embeddings_v2([self.ent_tot, self.dim], 'ent_im_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.rel_re_embeddings = init_embeddings_v2([self.rel_tot, self.dim], 'rel_re_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        #std = 1.0 / math.sqrt(100)
        #self.rel_im_embeddings = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.rel_tot, self.dim], stddev=std))
        self.rel_im_embeddings = init_embeddings_v2([self.rel_tot, self.dim], 'rel_im_embeddings',
                                                self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.ent_embeddings = init_embeddings_v2([self.ent_tot, self.dim*2], 'ent_embeddings',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        self.rel_embeddings = init_embeddings_v2([self.rel_tot, self.dim*2], 'rel_embeddings',
                                              self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
        """
        ss = tf.compat.v1.Session()
        ss.run(tf.compat.v1.global_variables_initializer())
        """

    def calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return -(tf.reduce_sum(h * r * t, 1)+
                 tf.reduce_sum(h_re * (r_re * t_re + r_im * t_im) + h_im * (r_re * t_im - r_im * t_re), 1))

    def get_score(self, h, r, t):
        sc_h, re_h, im_h = h[0], h[1], h[2]
        sc_t, re_t, im_t = t[0], t[1], t[2]
        sc_r, re_r, im_r = r[0], r[1], r[2]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return -tf.reduce_sum(tf.reshape(sc_h * sc_r, [b_size, 1, self.dim * 2]) * sc_t, 2) \
                   - tf.reduce_sum(tf.reshape(re_h * re_r - im_h * im_r, [b_size, 1, self.dim]) * re_t
                    + tf.reshape(re_h * im_r + im_h * re_r, [b_size, 1, self.dim]) * im_t, 2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction
            return -tf.reduce_sum(tf.reshape(sc_h * (sc_r * sc_t), [b_size, 1, self.dim * 2]), 2) \
                   - tf.reduce_sum(tf.reshape(re_h * (re_r * re_t + im_r * im_t), [b_size, 1, self.dim])
                    + tf.reshape(im_h * (re_r * im_t - im_r * re_t), [b_size, 1, self.dim]), 2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return -tf.reduce_sum(tf.reshape(sc_r * (sc_h * sc_t), [b_size, 1, self.dim * 2]), 2) \
                   -tf.reduce_sum(tf.reshape(re_r * (re_h * re_t + im_h * im_t), [b_size, 1, self.dim])
                   + tf.reshape(im_r * (re_h * im_t - im_h * re_t), [b_size, 1, self.dim]), 2)

    def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
        b_size = h_id.shape[0]

        sc_h = self.ent_embeddings(h_id)
        re_h = self.ent_re_embeddings(h_id)
        im_h = self.ent_im_embeddings(h_id)

        sc_t = self.ent_embeddings(t_id)
        re_t = self.ent_re_embeddings(t_id)
        im_t = self.ent_im_embeddings(t_id)

        sc_r = self.rel_embeddings(r_id)
        re_r = self.rel_re_embeddings(r_id)
        im_r = self.rel_im_embeddings(r_id)

        if mode == 'entities':
            sc_candidates = self.ent_embeddings
            sc_candidates = tf.reshape(sc_candidates, [1, self.ent_tot, -1])
            sc_candidates = tf.tile(sc_candidates, [b_size, 1, 1])

            re_candidates = self.ent_re_embeddings
            re_candidates = tf.reshape(re_candidates, [1, self.ent_tot, -1])
            re_candidates = tf.tile(re_candidates, [b_size, 1, 1])

            im_candidates = self.ent_im_embeddings
            im_candidates = tf.reshape(im_candidates, [1, self.ent_tot, self.dim])
            im_candidates = tf.tile(im_candidates, [b_size, 1, 1])

        else:
            sc_candidates = self.rel_embeddings
            sc_candidates = tf.reshape(sc_candidates, [1, self.rel_tot, self.dim * 2])
            sc_candidates = tf.tile(sc_candidates, [b_size, 1, 1])

            re_candidates = self.rel_re_embeddings
            re_candidates = tf.reshape(re_candidates, [1, self.rel_tot, self.dim])
            re_candidates = tf.tile(re_candidates, [b_size, 1, 1])

            im_candidates = self.rel_im_embeddings
            im_candidates = tf.reshape(im_candidates, [1, self.rel_tot, self.dim])
            im_candidates = tf.tile(im_candidates, [b_size, 1, 1])

        return (sc_h, re_h, im_h), \
               (sc_r, re_r, im_r), \
               (sc_t, re_t, im_t), \
               (sc_candidates, re_candidates, im_candidates)

    def call(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        h = self.ent_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        score = self.calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
        return score

    def save(self):
        ent_embeds = self.ent_embeddings.numpy() if self.ent_embeddings is not None else None
        rel_embeds = self.rel_embeddings.numpy() if self.rel_embeddings is not None else None
        read.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=None)
        read.save_special_embeddings(self.out_folder, 'ent_re_embeddings', '', self.ent_re_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'ent_im_embeddings', '', self.ent_im_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'rel_re_embeddings', '', self.rel_re_embeddings.numpy(), None)
        read.save_special_embeddings(self.out_folder, 'rel_im_embeddings', '', self.rel_im_embeddings.numpy(), None)
