import pandas as pd
import numpy as np
import sys

np.random.seed(11)

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
        (64, (1, 1)),
        (128, (2, 2)),
        *[(128, (1, 1)) for _ in range(3)],
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
weights = np.load(sys.argv[1], allow_pickle = True)['arr_0'].tolist()
test = np.array([i.split(' ') for i in pd.read_csv(sys.argv[2])['feature']]).astype('float')
test = (test/255.).reshape((-1, 48, 48, 1))
model.set_weights(weights)
predict = model.predict(test)
ans = (predict.argmax(axis=1)[:,None] == range(predict.shape[1])).astype(int)
ans = ans.argmax(axis = 1)

with open(sys.argv[3], 'w') as f:
    f.write('id,label\n')
    for i in range(len(ans)):
        f.write(str(i) + ',' + str(ans[i]) + '\n')

