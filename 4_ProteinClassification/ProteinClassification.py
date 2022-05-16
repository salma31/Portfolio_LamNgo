
#_______________________________SETUP__________________________________


import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from Bio import SeqIO

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, GlobalMaxPooling1D, LSTM
import sys
import keras.backend as K
from sklearn.metrics import f1_score


#tf.debugging.set_log_device_placement(True)
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


# ________________________________DATA PREPROCESSING________________________________________

df_train = pd.DataFrame(columns = ["seq", "class"])
df_val = pd.DataFrame(columns = ["seq", "class"])
df_test = pd.DataFrame(columns = ["seq"])

arg_dict ={"aminoglycoside": 0, "macrolide-lincosamide-streptogramin": 1, "polymyxin": 2,
"fosfomycin": 3, "trimethoprim": 4, "bacitracin": 5, "quinolone": 6, "multidrug": 7,
"chloramphenicol": 8, "tetracycline": 9, "rifampin": 10, "beta_lactam": 11,
"sulfonamide": 12, "glycopeptide": 13, "nonarg": 14}

for seq_record in SeqIO.parse("train.fasta", "fasta"):
    seq_id = seq_record.id.split("|")
    seq = str(seq_record.seq).replace(",", "").replace("(", "").replace(")", "")

    if seq_id[0] == "sp":
        #rec = {"seq": [seq], "class": [0]}
        rec = {"seq": [seq], "class": [14]}
        df_train = pd.concat([df_train, pd.DataFrame(rec)], ignore_index=True)
        #df_train = df_train.append(rec, ignore_index=True)
    else:
        try:
            rec = {"seq": [seq], "class": [arg_dict[seq_id[3]]]}
            #rec = {"seq": [seq], "class": [1]}
            #print(df_train.shape)
            #print(pd.DataFrame(rec.values(), columns=df_train.columns))            
            df_train = pd.concat([df_train, pd.DataFrame(rec)], ignore_index=True)

            #df_train = df_train.append(rec, ignore_index=True)
        except KeyError:
            print(seq_record.id)
            
for seq_record in SeqIO.parse("val.fasta", "fasta"):
    seq_id = seq_record.id.split("|")
    seq = str(seq_record.seq).replace(",", "").replace("(", "").replace(")", "")

    if seq_id[0] == "sp":
        #rec = {"seq": [seq], "class": [0]}
        rec = {"seq": [seq], "class": [14]}
        df_val = pd.concat([df_val, pd.DataFrame(rec)], ignore_index=True)
        #df_val = df_val.append(rec, ignore_index=True)
    else:
        try:
            #rec = {"seq": [seq], "class": [1]}
            rec = {"seq": [seq], "class": [arg_dict[seq_id[3]]]}
            df_val = pd.concat([df_val, pd.DataFrame(rec)], ignore_index=True)
            #df_val = df_val.append(rec, ignore_index=True)
        except KeyError:
            print(seq_record.id)            
            
for seq_record in SeqIO.parse("test.fasta", "fasta"):
    seq_id = seq_record.id.split("|")
    seq = str(seq_record.seq).replace(",", "").replace("(", "").replace(")", "")
    rec = {"seq": [seq], "id": [seq_id]}
    #df_test = df_test.append(rec, ignore_index = True)
    df_test = pd.concat([df_test, pd.DataFrame(rec)], ignore_index=True)

    
   
classes = df_train['class'].value_counts()[:15].index.tolist()
train_sm = df_train.loc[df_train['class'].isin(classes)].reset_index()
val_sm = df_val.loc[df_val['class'].isin(classes)].reset_index()
test_sm = df_test

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
  char_dict = {}
  for index, val in enumerate(codes):
    char_dict[val] = index+1

  return char_dict

char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))

def integer_encoding(data):
  """
  - Encodes code sequence to integer values.
  - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
  """
  
  encode_list = []
  for row in data['seq'].values:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list


train_encode = integer_encoding(train_sm) 
val_encode = integer_encoding(val_sm) 
test_encode = integer_encoding(test_sm) 


max_length = 1576
train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

print(train_pad.shape, val_pad.shape, test_pad.shape)

train_ohe = to_categorical(train_pad)
val_ohe = to_categorical(val_pad)
test_ohe = to_categorical(test_pad)

print(train_ohe.shape, test_ohe.shape, test_ohe.shape)

le = LabelEncoder()
y_train_le = le.fit_transform(train_sm['class'])
y_val_le = le.transform(val_sm['class'])
print(y_train_le.shape, y_val_le.shape)

y_train = to_categorical(y_train_le)
y_val = to_categorical(y_val_le)
print(y_train.shape, y_val.shape)


def display_model_score(model, train, val, batch_size):

  train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
  print('Train loss: ', train_score[0])
  print('Train accuracy: ', train_score[1])
  print('-'*70)

  val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
  print('Val loss: ', val_score[0])
  print('Val accuracy: ', val_score[1])
  print('-'*70)


# ____________________________________MODEL ARCHITECTURE______________________________________________________  
# architecture from https://towardsdatascience.com/protein-sequence-classification-99c80d0ad2df  
def residual_block(data, filters, d_rate):
  """
  _data: input
  _filters: convolution filters
  _d_rate: dilation rate
  """

  shortcut = data

  bn1 = BatchNormalization()(data)
  act1 = Activation('relu')(bn1)
  conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)

  #bottleneck convolution
  bn2 = BatchNormalization()(conv1)
  act2 = Activation('relu')(bn2)
  conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)

  #skip connection
  x = Add()([conv2, shortcut])

  return x
  
  
x_input = Input(shape=(1576, 21))

#initial conv
conv = Conv1D(512, 1, padding='same')(x_input) 

# per-residue representation
res1 = residual_block(conv, 512, 2)
res2 = residual_block(res1, 512, 3)

x = MaxPooling1D(3)(res2)
x = Dropout(0.8)(x)

# softmax classifier
x = Flatten()(x)
#x = Dense(128, activation='sigmoid')(x)
x_output = Dense(15, activation='softmax', kernel_regularizer=l2(0.0001))(x)

model2 = Model(inputs=x_input, outputs=x_output)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model2.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])

model2.summary()
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
mcp_save = ModelCheckpoint('best.h5', save_best_only=True, monitor='val_loss', mode='min')


#___________________________________________ TRAINING ______________________________________________
history2 = model2.fit(
    train_ohe, y_train,
    epochs=100, batch_size=256,
    validation_data=(val_ohe, y_val),
    callbacks = [mcp_save]
    #callbacks=[es]
    )


model2.save_weights('/content/gdrive/MyDrive/A2/model8.h5')















