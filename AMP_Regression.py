import warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from codes.BINARY import *
from codes.AAINDEX import *
from codes.BLOSUM62 import *
from codes.ZSCALE import *
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,GlobalAveragePooling2D,BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Conv1D,MaxPooling1D
from keras.layers import LSTM, Dense, Dropout, Activation, GRU, SimpleRNN
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

my_seed =6
tf.random.set_seed(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)
os.environ['PYTHONHASHSEED'] = str(my_seed)

file_path = r"train.fasta"
f = open(file_path, 'r', encoding='utf-8')
label = []

def spearmanr(y_true, y_pred):
    diff_pred, diff_true = y_pred - np.mean(y_pred), y_true - np.mean(y_true)
    scalar = np.sum(diff_pred * diff_true) / np.sqrt(np.sum(diff_pred **2) * np.sum(diff_true **2))
    return scalar

def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        input_data = input_data+pad_token
        result = input_data
    return result

fasta_list = np.array(f.readlines())
aa_feature_list = []
all_feature = []
len_seq_max = 30
for flag in range(0, len(fasta_list), 2):
    if "," in fasta_list[flag+1].strip('\n').strip():
        seq = fasta_list[flag+1].strip('\n').strip()
    elif "X" in fasta_list[flag+1].strip('\n').strip():
        seq = fasta_list[flag+1].strip('\n').strip()
    else:
        fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
        if len(fasta_str[0][1]) > len_seq_max:
            fasta_str[0][1] = fasta_str[0][1][0:len_seq_max]
        else:
            fasta_str[0][1] = pad_to_length(fasta_str[0][1], "-", len_seq_max)
        bin_output = BINARY(fasta_str)
        aai_output = AAINDEX(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)
        feature_id = bin_output[1][0].split('>')[1]
        bin_output[1].remove(bin_output[1][0])
        aai_output[1].remove(aai_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])
        bin_feature = []
        aai_feature = []
        blo_feature = []
        zsl_feature = []
        for i in range(0, len(bin_output[1]), 20):
            temp = bin_output[1][i:i + 20]
            bin_feature.append(temp)
        for i in range(0, len(aai_output[1]), 531):
            temp = [float(i) for i in aai_output[1][i:i + 531]]
            aai_feature.append(temp)
        for i in range(0, len(blo_output[1]), 20):
            temp = blo_output[1][i:i + 20]
            blo_feature.append(temp)
        for i in range(0, len(zsl_output[1]), 5):
            temp = zsl_output[1][i:i + 5]
            zsl_feature.append(temp)
        aa_fea_matrx = np.hstack([np.array(bin_feature), np.array(blo_feature), np.array(zsl_feature)])
        aa_feature_list.append(aa_fea_matrx)
        label.append(feature_id)
aa_feature_list = np.array(aa_feature_list)

timesteps = 1
data_dim = len_seq_max*45
x = aa_feature_list
x = np.reshape(x, (len(x), timesteps, data_dim))
x = np.array(x)

y = label
y = [float(lab) for lab in y]
y = np.array(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)

model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(timesteps, data_dim), name='lstm1'))
model.add(Dropout(0.6, name='dropout'))
model.add(Dense(1, Activation('linear')))
model.summary()

print('Compiling the Model...')
optimizer =tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

print("Train...")

checkpoint = ModelCheckpoint("best_model.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

history = model.fit(x=x_train, y=y_train,
                    validation_data=(x_val, y_val),
                    epochs=50,
                    batch_size=8,
                    callbacks=[checkpoint])

model.load_weights("best_model.h5")

probas = []
proba = model.predict(x_val)
for i in proba:
    probas.append(i[0])
probas = np.array(probas)
score = spearmanr(probas, y_val)
pcc = pearsonr(probas, y_val)
print(score)

r2 = r2_score(y_val, probas)
print(f"R² Score: {r2}")

plt.figure(figsize=(4, 4))
plt.scatter(y_val, probas, alpha=1, c=(120 / 255, 208 / 255, 203 / 255), s=80)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '--', color='pink')
# plt.ylim(-1,3)
# plt.xlim(-1,3)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("R² Correlation between True and Predicted Values")
plt.legend()
plt.show()
