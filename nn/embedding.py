# -*- coding:utf-8 -*-

__author__ = 'ghostviper'

from config import SAMPLE_LENGTH
import numpy as np


def build_embedding_matrix(df, labels, word2vec_model):
    """
    :param df: pandas dateframe df[['content', 'labels']],labels用\n进行分割
    :param labels: 所有的分类label列表
    :param word2vec_model: gensim格式的word2vec模型
    :return: embeddeding后的x和one-hot encoding后的y
    """
    embedding_size = word2vec_model.vector_size
    doc_count = df.counts()
    label_count = len(labels)
    matrix_x = np.zeros(shape=(doc_count, SAMPLE_LENGTH, embedding_size))
    matrix_y = np.zeros(shape=(doc_count, label_count), dtype=np.bool_)
    for k, row in df.iterrows():
        words = row['content'].split()[SAMPLE_LENGTH:]
        for i, w in enumerate(words):
            if w in word2vec_model.wv:
                word_vector = word2vec_model.wv[w].reshape(1, -1)
                matrix_x[k][i] = word_vector
            # 如果词没出现在word2vec中则默认为0向量
        for label in row['labels'].split("\n"):
            matrix_y[k][labels.index(label)] = True
    return matrix_x, matrix_y


