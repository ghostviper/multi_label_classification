# -*- coding:utf-8 -*-

__author__ = 'ghostviper'

from keras.layers import (
    Input, Dense, Dropout,
    BatchNormalization, MaxPooling1D,
    Conv1D, Flatten,  Concatenate
)
from keras.models import Model
from keras import optimizers
from config import (
    SAMPLE_LENGTH,
    DROPOUT_RATE,
    NGRAM_LENGTH,
    NB_FILTER
)


def cnn(embedding_size, output_length):
    """
    text-cnn模型
    :param embedding_size: 词嵌入的维度（一个词语通过word2vec提取出的特征维度）
    :param output_length: 分类的数目
    :return:
    """
    inputs, conv_layers = [], []
    for ngram in NGRAM_LENGTH:
        current_input = Input(shape=(SAMPLE_LENGTH, embedding_size))
        inputs.append(current_input)
        # 卷积
        conv = Conv1D(
            filters=NB_FILTER,
            kernel_size=ngram,
            strides=1,
            kernel_initializer='lecun_uniform',
            activation='relu'
        )(current_input)
        # batch normalization
        bm = BatchNormalization()(conv)
        # 池化计算
        # new_height = (input_height - filter_height) / strides + 1
        # new_width = (input_width - filter_width) / strides + 1
        pool_size = (SAMPLE_LENGTH - ngram)/1 + 1
        maxpooling = MaxPooling1D(pool_size=pool_size)(bm)
        conv_layers.append(maxpooling)
    # 将每个ngram的卷积层拼接起来
    merged_layer = Concatenate()(conv_layers)
    # dropout防止过拟合
    dropout = Dropout(DROPOUT_RATE)(merged_layer)
    # 展平
    flatten = Flatten()(dropout)
    # 构建分类层
    dense = Dense(1024, activation='relu')(flatten)
    outputs = Dense(output_length, activation='softmax')(dense)
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    # 编译模型 三个指标 optimizer、loss、metrics
    optimizer = optimizers.Adam(lr=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'top_k_catrgorical_accuracy']
    )
    return model



