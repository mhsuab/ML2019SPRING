import pandas as pd
import numpy as np
import os
import sys
import jieba
import emoji
from gensim.models import Word2Vec

from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

from multiprocessing import Pool

jieba_dict = sys.argv[4]
embed_size = 128
batch_size = 128
max_length = 36
iteration = 100
model_type = 2

jieba.set_dictionary(jieba_dict)

np.random.seed(36)

def split_valid_set(x, y, percentage = 0.8, shuffle = True):
    n = len(x)
    valid_data_size = int(np.floor(n * percentage))
    if shuffle:
        x, y = myShuffle(x, y)
    #xtrain, ytrain, xvalid, yvalid
    return x[0:valid_data_size], y[0:valid_data_size], x[valid_data_size:], y[valid_data_size:]

def myShuffle(x, y):
    r = np.arange(len(x))
    np.random.shuffle(r)
    return x[r], y[r]

def processJieba(x):
    return list(map(emoji.demojize, jieba.lcut(str(x[0]))))

def processJieban(x):
    tmp = []
    for s in x:
        ss = jieba.lcut(str(s[0]))
        ss = list(map(emoji.demojize, ss))
        tmp.append(ss)
    return tmp

train_x = pd.read_csv(sys.argv[1], usecols = ['comment']).values[:119018]
test_x = pd.read_csv(sys.argv[3], usecols = ['comment']).values
train_y = pd.read_csv(sys.argv[2], usecols = ['label']).values[:119018]
P = Pool()
train_x = P.map(processJieba, train_x)
P.close()
P.join()

print ('???????')

P = Pool()
test_x = P.map(processJieba, test_x)
P.close()
P.join()
'''
train_x = processJieba(train_x)
test_x = processJieba(test_x)
'''

if os.path.exists('word2vec.bin'):
    embed = Word2Vec.load('word2vec.bin')

else:
    print ('embed')
    embed = Word2Vec(train_x + test_x, size = embed_size, workers = 4, window = 5, min_count = 1, iter = iteration)
    embed.save('word2vec.bin')

def sen2vec(sen):
    sen = sen[:max_length]
    sen = list(map(lambda w: embed.wv[w], sen))
    sen += [np.zeros(embed.wv.vector_size)] * (max_length - len(sen))
    return sen

P = Pool()
train_x = P.map(sen2vec, train_x)
P.close()
P.join()

P = Pool()
test_x = P.map(sen2vec, test_x)
P.close()
P.join()

train_x = np.array(train_x)
test_x = np.array(test_x)

np.save('train_x_em', train_x)
np.save('test_x_em', test_x)
np.save('train_y', train_y)

xtrain, ytrain, xvalid, yvalid = split_valid_set(train_x, train_y, 0.8, True)

model = Sequential()
#model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights = [pretrained_weight]))

if model_type == 2:
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
    model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.4)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 0:
    model.add(LSTM(units = 128, return_sequences = False))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units = 1, activation = 'sigmoid'))

elif model_type == 1:
    model.add(LSTM(512,return_sequences=True))
    model.add(LSTM(256,return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 3:
    model.add(GRU(30, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = False, dropout = 0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 4:
    model.add(GRU(128, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = True, dropout = 0.5))
    model.add(GRU(64, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = True, dropout = 0.5))
    model.add(GRU(32, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = False, dropout = 0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 5:
    model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 6:
    model.add(GRU(30, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = False, dropout = 0.5))
    model.add(Dense(1, activation='relu'))

elif model_type == 7:
    model.add(LSTM(256, activation='tanh', dropout=0.2, return_sequences=True, kernel_initializer = 'Orthogonal'))
    model.add(LSTM(128, activation='tanh', dropout=0.2, return_sequences=True, kernel_initializer = 'Orthogonal'))
    model.add(LSTM(128, activation='tanh', dropout=0.2, return_sequences=False, kernel_initializer = 'Orthogonal'))
    model.add(Dense(1, activation = 'sigmoid'))

elif model_type == 8:
    model.add(GRU(30, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = False, dropout = 0.5))
    model.add(Dense(1, activation='sigmoid'))

elif model_type == 9:
    model.add(GRU(128, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = True, dropout = 0.5))
    model.add(GRU(64, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = True, dropout = 0.5))
    model.add(GRU(32, activation = 'tanh', recurrent_activation = 'hard_sigmoid', return_sequences = False, dropout = 0.5))
    model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max'), 
            ModelCheckpoint('train.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            TerminateOnNaN(),
            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patiece = 5, min_lr = 0.001),
            CSVLogger('train_acc.csv'.format(model_type), separator = ',', append = False)]

print('Train...')

model.fit(xtrain, ytrain, batch_size = batch_size, epochs = 100, validation_data = [xvalid, yvalid], callbacks = callbacks, shuffle = True)
model.summary()

#model = load_model('models/pre_stack_{}.h5'.format(model_type))
model = load_model('train.h5'.format(model_type))
predict = model.predict(test_x)
ans = (predict >= 0.5).astype(int)

#with open('result/pre_stack_{}.csv'.format(model_type), 'w') as f:
with open('ans.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(ans)):
        f.write(str(i) + ',' + str(ans[i][0]) + '\n')




