import pandas as pd
import numpy as np
import sys

np.random.seed(11)

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
raw = pd.read_csv(sys.argv[1])
x = np.array([i.split(' ') for i in raw['feature']]).astype('float')
y = pd.get_dummies(raw['label']).values
x = (x/255.).reshape((-1, 48, 48, 1))

xtrain, ytrain, xvalid, yvalid = split_valid_set(x, y, 0.8)


from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras.activations import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

n_classes = 7
input_shape = (48, 48, 1)
alpha = 1.0

def conv_block(tensor, channels, strides, alpha):
    x = Conv2D(int(channels * alpha), kernel_size = (3, 3),
                strides = strides, use_bias = False, padding = 'same')(tensor)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 1./20.)(x)
    return x

def dw_block(tensor, channels, strides, alpha):
    x = DepthwiseConv2D(kernel_size = (3, 3), strides = strides,
                        use_bias = False, padding = 'same')(tensor)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 1./20.)(x)
    x = Conv2D(int(channels * alpha), kernel_size = (1, 1), 
                use_bias = False, padding = 'same')(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha = 1./20.)(x)
    return x

x_in = Input(shape = input_shape)
x = conv_block(x_in, 32, (2, 2), alpha = alpha)
layers = [
        #(32, (1, 1)),
        #(64, (2, 2)),
        (64, (1, 1)),
        (64, (1, 1)),
        (128, (2, 2)),
        *[(128, (1, 1)) for _ in range(2)],
        #(256, (2, 2)),
        #(256, (1, 1)),
        #(512, (2, 2)),
        #*[(512, (1, 1)) for _ in range(1)],
        #(1024, (2, 2)),
        #(1024, (2, 2))
        ]

for i, (j, k) in enumerate(layers):
    x = dw_block(x, j, k, alpha = alpha)

x = GlobalAvgPool2D()(x)
#x = Dense(16, activation = 'relu')(x)
x = Dense(7, activation = 'softmax')(x)

model = Model(inputs = x_in, outputs = x)
print (model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
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

callbacks = [EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='max'), 
            ModelCheckpoint('{}.hdf5'.format(sys.argv[2]), monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only = True),
            TerminateOnNaN(),
            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patiece = 5, min_lr = 0.001),
            CSVLogger('{}.csv'.format(sys.argv[2]), separator = ',', append = False)]

model.fit_generator(datagen.flow(xtrain, ytrain, batch_size = 32),
    steps_per_epoch = xtrain.shape[0]//32,
    validation_data = (xvalid, yvalid),
    epochs = 500, verbose = 1, max_q_size = 100,
    callbacks = callbacks)

