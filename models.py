import numpy as np
from sklearn.metrics import accuracy_score
from glob import glob
from sklearn.model_selection import KFold

from keras.utils.np_utils import to_categorical
from keras.layers import (
    Input,
    Conv1D,
    MaxPool1D,
    Flatten,
    Dense,
    Dropout,
    Embedding,
    Activation,
    BatchNormalization,
    Concatenate,
    SimpleRNN
)
from keras.models import Model


def stf_rnn(nb_points, emb_size1, tm_length, emb_size2, window_size, rnn_size):

    s_input = Input((window_size, ), dtype='int32', name='S')  # 空间特征
    t_input = Input((window_size, ), dtype='int32', name='T')  # 时间特征

    emb1 = Embedding(nb_points + 1, emb_size1)
    emb2 = Embedding(tm_length + 1, emb_size2)

    xe = emb1(s_input) 
    he = emb2(t_input) 

    x = Concatenate()([xe, he])  # 拼接embedding之后的空间特征和时间特征
    x = SimpleRNN(rnn_size)(x)   # 将上一步拼接的结果输入进Rnn中
    y = Dense(nb_points, activation='softmax')(x)  # 将上一步结果输入进全连接层

    model = Model([s_input, t_input], y)

    # 激活函数 adadelta
    # 损失函数 categorical_crossentropy
    # 评价指标 accuracy
    model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])

    return model


