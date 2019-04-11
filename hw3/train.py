import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
KTF.set_session(sess)
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config = config)
KTF.set_session(session)


def myShuffle(x, y):
    r = np.arange(len(x))
    np.random.shuffle(r)
    return x[r], y[r]

def split_valid_set(x, y, percentage = 0.8):
    n = len(x)
    valid_data_size = int(np.floor(n * percentage))
    x, y = myShuffle(x, y)
    #xtrain, ytrain, xvalid, yvalid
    return x[0:valid_data_size], y[0:valid_data_size], x[valid_data_size:], y[valid_data_size:]

#read file

raw = pd.read_csv('data/train.csv')
x = np.array([i.split(' ') for i in raw['feature']]).astype('float')
#y = pd.get_dummies(raw['label']).values
x = (x).reshape((-1, 48, 48, 1))

test = np.array([i.split(' ') for i in pd.read_csv('data/test.csv')['feature']]).astype('float')
test = (test).reshape((-1, 48, 48, 1))


#x = np.load('data/xtrain.npy')
y = np.load('data/ytrain.npy')
#test = np.load('data/xtest.npy')

xtrain, ytrain, xvalid, yvalid = split_valid_set(x, y, 0.9)

from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LeakyReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, History, CSVLogger

model = Sequential()

model.add(Conv2D(64, input_shape = xtrain[0].shape, kernel_size = (5, 5), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(LeakyReLU(alpha = 1./20.))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(LeakyReLU(alpha = 1./20.))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(Dropout(0.35))

model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(LeakyReLU(alpha = 1./20.))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(Dropout(0.35))

model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
model.add(LeakyReLU(alpha = 1./20.))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(512, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax', kernel_initializer = 'glorot_normal'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
callbacks = []
callbacks.append(ModelCheckpoint('models/' + sys.argv[1] + '.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
#callbacks.append(CSVLogger('csvlogger/' + sys.argv[1] + '.csv', separator = ',', append = False))
callbacks.append(History())

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = [0.8, 1.2],
    horizontal_flip = True,
    vertical_flip = False)

datagen.fit(xtrain)

history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size = 32),
    steps_per_epoch = xtrain.shape[0]//32,
    validation_data = (xvalid, yvalid),
    epochs = 500, verbose = 1, max_q_size = 100,
    callbacks = callbacks)


predict = model.predict(test)
testy = np.argmax(predict, axis = 1)
with open(sys.argv[1] + '.csv', 'w') as f:
    f.write('id,label\n')
    for i in range(len(testy)):
        f.write(str(i) + ',' + str(testy[i]) + '\n')





