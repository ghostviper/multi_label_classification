# text-cnn 的 keras实现

本项目主要使用 keras 实现了textcnn 的基本结构和最简单实现，语料数据的收集、预处理、分词过滤需要自己实现, 主要文件包含:

- *nn/embedding.py* 实现了基本的基于word2vec的词嵌入

- *nn/models.py* 实现了textcnn 模型

- main.py 调用模型训练：

# 基本解释

*main.py* 的输入参数：

- 预处理完毕的pandas dataframe文件，需要包含字段 'content', 'labels'

- 训练好的word2vec模型

*config.py* 参数：

- SAMPLE_LENGTH 样本文件选取的前几个词
- DROPOUT_RATE dropout率
- NGRAM_LENGTH ngram大小列表（按照ngram个词进行卷积）
- NB_FILTER 每个卷积层输出的通道数(特征图数目)
- BATCH_SIZE 输入迭代的批次大小
- EPOCHS 迭代的轮次

# 备注
- 1.词嵌入模型可以选择使用word2vec, ELMo, GloVe, [Bert](https://github.com/google-research/bert), [腾讯AI Lab](https://ai.tencent.com/ailab/nlp/embedding.html),目前word2vec基于维基百科的语料基本能满足要求
- 2.multi classification使用激活函数 *softmax*、multi label classification使用激活函数 *sigmoid*
- 3.预处理的分词器如果选择jieba，则务必构建领域相关的自定义词典
- 4.本项目可以改造实现为多分类和多标签分类：基本可以解决类似情感分析、敏感检测、文本分类的问题

# 参考
- 基于tensorflow的实现可以参看项目[some_practice_for_text_classification](https://github.com/ghostviper/some_practice_for_text_classification)
- 关于如何训练word2vec模型和使用可以参考 [中英文维基百科语料上的Word2Vec实验](https://blog.csdn.net/yangyangrenren/article/details/56286394)
