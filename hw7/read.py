import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread

def load_image(filename):
    return np.asarray(imread(filename), dtype = 'int32')
    #return np.asarray(Image.open(filename), dtype = 'int32')

def read(path, bool_S = True):
    test = []
    for i in range(40000):
        test.append(load_image(os.path.join(path, f'{(i + 1):06}.jpg')))
    test = np.array(test)
    if bool_S:
        np.save('images', test)
    return test

def autoencoder_d():
    input_img = Input(shape = (3072,))
    encoded = Dense(128, activation = 'relu')(input_img)
    encoded = Dense(64, activation = 'relu')(encoded)
    encoded = Dense(32, activation = 'relu')(encoded)

    decoded = Dense(64, activation = 'relu')(encoded)
    decoded = Dense(128, activation = 'relu')(decoded)
    decoded = Dense(3072, activation = 'sigmoid')(decoded)
    return Model(input_img, encoded), Model(input_img, decoded)

def autoencoder():
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    return Model(input_img, encoded), Model(input_img, decoded)

def auto_encoder(xtrain, datagen = False, file_name = 'datagen_', epoch = 100):
    input_img = Input(shape = (32, 32, 3))
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha = 0.02)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha = 0.02)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha = 0.02)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = LeakyReLU(alpha = 0.02)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha = 0.02)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha = 0.02)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), padding = 'same', activation = 'sigmoid')(x)

    encoder = Model(input_img, encoded)
    auto = Model(input_img, decoded)
    auto.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    auto.summary()
    if datagen:
        Datagen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            rotation_range = 30,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            zoom_range = [0.8, 1.2],
            horizontal_flip = True,
            vertical_flip = False)
        Datagen.fit(xtrain)
        auto.fit_generator(Datagen.flow(xtrain, xtrain, batch_size = 32),
                    steps_per_epoch = xtrain.shape[0]//32,
                    epochs = epoch, verbose = 1, max_q_size = 100)
    else:
        auto.fit(xtrain, xtrain, batch_size = 256, epochs = epoch)
    encoder.save('{}_encoder_{}.h5'.format(file_name, epoch))
    auto.save('{}_auto_{}.h5'.format(file_name, epoch))

    return encoder, auto