# -*- coding:utf-8 -*-

__author__ = 'ghostviper'

from config import BATCH_SIZE, EPOCHS
from sklearn.model_selection import train_test_split
from nn.models import cnn
from nn.embedding import build_embedding_matrix


def train(df, labels, word2vec_model):
    # 对语料进行词嵌入
    x, y = build_embedding_matrix(df=df, labels=labels, word2vec_model=word2vec_model)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 建立模型
    model = cnn(
        embedding_size=word2vec_model.vector_size,
        output_length=len(labels)
    )
    # 训练
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
