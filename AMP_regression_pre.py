# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 4:08
# @Author  : Cheng Ge
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Bio.SeqUtils import molecular_weight
from tensorflow.keras.utils import to_categorical
import math
from codes.BINARY import *
from codes.AAINDEX import *
from codes.BLOSUM62 import *
from codes.ZSCALE import *
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
# ===================================================特征提取================================================================
file_path = r"D:/y3/hlh/test_new.fasta"
f = open(file_path, 'r', encoding='utf-8')
label = []

def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    # input_data = list(input_data)
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        input_data = input_data+pad_token
        result = input_data
    return result

# 获取序列特征
fasta_list = np.array(f.readlines())
aa_feature_list = []
all_feature = []
len_seq_max = 30
sequence = []
for flag in range(0, len(fasta_list)):
    if "," in fasta_list[flag].strip('\n').strip():
        seq = fasta_list[flag].strip('\n').strip()
        print(f"第{flag}个数据错误,序列{seq}包含,")
    elif "X" in fasta_list[flag].strip('\n').strip():
        seq = fasta_list[flag].strip('\n').strip()
        print(f"第{flag}个数据错误,序列{seq}包含X")
    else:
        seq = fasta_list[flag].strip('\n').strip()

        sequence.append(seq)
        fasta_str = [[">1", fasta_list[flag].strip('\n').strip()]]
        if len(fasta_str[0][1]) > len_seq_max:  # acp240最低23，acp740最低33
            fasta_str[0][1] = fasta_str[0][1][0:len_seq_max]
        else:
            fasta_str[0][1] = pad_to_length(fasta_str[0][1], "-", len_seq_max)
        bin_output = BINARY(fasta_str)
        # aai_output = AAINDEX(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)
        bin_output[1].remove(bin_output[1][0])
        # aai_output[1].remove(aai_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])
        bin_feature = []
        # aai_feature = []
        blo_feature = []
        zsl_feature = []
        for i in range(0, len(bin_output[1]), 20):
            temp = bin_output[1][i:i + 20]
            bin_feature.append(temp)
        # for i in range(0, len(aai_output[1]), 531):
        #     temp = [float(i) for i in aai_output[1][i:i + 531]]
        #     aai_feature.append(temp)
        for i in range(0, len(blo_output[1]), 20):
            temp = blo_output[1][i:i + 20]
            blo_feature.append(temp)
        for i in range(0, len(zsl_output[1]), 5):
            temp = zsl_output[1][i:i + 5]
            zsl_feature.append(temp)
        aa_fea_matrx = np.hstack([np.array(bin_feature), np.array(blo_feature), np.array(zsl_feature)])
        aa_feature_list.append(aa_fea_matrx)
aa_feature_list = np.array(aa_feature_list)
#  =======================================Dataset=============================================
timesteps = 1
data_dim = len_seq_max*45
x = aa_feature_list
x_train = np.reshape(x, (len(x), timesteps, data_dim))
x_train = np.array(x_train)

model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(timesteps, data_dim)))
model.add(Dropout(0.6, name='dropout'))
model.add(Dense(1, Activation('linear')))
model.summary()

model.load_weights(filepath=r"D:/中国海洋大学课题/抗菌肽生成/代码/活性值预测模型/hlh_v1.h5", by_name=False, skip_mismatch=False, options=None)

proba = model.predict(x_train)
k=-1
for i in proba:
    k += 1
    output = 0.001*(math.pow(10, i[0]))*molecular_weight(sequence[k],"protein")
    print(i[0])
