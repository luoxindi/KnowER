import math
import random

import numpy as np
#tf.disable_eager_execution()  #关闭eager运算
#tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
import tensorflow as tf
from sklearn import preprocessing


def init_embeddings(shape, name, init, is_l2_norm, dtype=tf.float32):
    embeds = None
    if init == 'xavier':
        embeds = xavier_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'normal':
        embeds = truncated_normal_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'uniform':
        embeds = random_uniform_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'unit':
        embeds = random_unit_init(shape, name, is_l2_norm, dtype=dtype)
    return embeds


def xavier_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    #tf.disable_eager_execution()  #关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('xavier_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.keras.initializers.glorot_normal())
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def truncated_normal_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    #tf.disable_eager_execution()  #关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('truncated_normal'):
        std = 1.0 / math.sqrt(shape[1])
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.truncated_normal(stddev=std))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_uniform_init(shape, name, is_l2_norm, minval=0, maxval=None, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('random_uniform'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.random_uniform(minval=minval, maxval=maxval))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_unit_init(shape, name, is_l2_norm, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('random_unit_init'):
        vectors = list()
        for i in range(shape[0]):
            vectors.append([random.gauss(0, 1) for j in range(shape[1])])
    embeddings = tf.Variable(preprocessing.normalize(np.matrix(vectors)), name=name, dtype=dtype)
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def orthogonal_init(shape, name, dtype=None):
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_eager_execution()  # 关闭eager运算
    # tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
    with tf.name_scope('orthogonal_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.initializers.orthogonal())
    return embeddings


def xavier_init_v2(shape, name, is_l2_norm, dtype):
    import tensorflow as tf
    embeddings = tf.keras.layers.Embedding(
        input_dim=shape[0], output_dim=shape[1], name=name,
        embeddings_initializer=tf.keras.initializers.glorot_normal(), )
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def truncated_normal_init_v2(shape, name, is_l2_norm, dtype):
    import tensorflow as tf
    std = 1.0 / math.sqrt(shape[1])
    embeddings = tf.Variable(tf.initializers.truncated_normal())
    embeddings = tf.keras.layers.Embedding(
        input_dim=shape[0], output_dim=shape[1], name=name,
        embeddings_initializer=tf.keras.initializers.truncated_normal(stddev=std), )
    #embeddings = tf.Variable(initial_value=tf.random.truncated_normal(shape=shape, stddev=std), name=name)
    #print(embeddings.get_weights())
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_uniform_init_v2(shape, name, is_l2_norm,  minval=0, maxval=None, dtype=None):
    import tensorflow as tf
    embeddings = tf.keras.layers.Embedding(
        input_dim=shape[0], output_dim=shape[1], name=name,
        embeddings_initializer=tf.keras.initializers.random_uniform(minval=minval, maxval=maxval), )
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_unit_init_v2(shape, name, is_l2_norm, dtype):
    import tensorflow as tf
    vectors = list()
    for i in range(shape[0]):
        vectors.append([random.gauss(0, 1) for j in range(shape[1])])
    embeddings = tf.compat.v1.get_variable(preprocessing.normalize(np.matrix(vectors)), name=name, dtype=dtype)
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def init_embeddings_v2(shape, name, init, is_l2_norm, dtype=tf.float32):
    embeds = None
    if init == 'xavier':
        embeds = xavier_init_v2(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'normal':
        embeds = truncated_normal_init_v2(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'uniform':
        embeds = random_uniform_init_v2(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'unit':
        embeds = random_unit_init_v2(shape, name, is_l2_norm, dtype=dtype)
    return embeds
