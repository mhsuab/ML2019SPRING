import numpy as np
import pandas as pd
import csv

def readTrainData(xpath = 'data/X_train', ypath = 'data/Y_train'):
    x = pd.read_csv(xpath).values
    x = np.array([np.append(x[i], [1.]) for i in range(len(x))])
    y = pd.read_csv(ypath).values
    return x, y.flatten()

def readTestData(path = 'data/X_test'):
    x = pd.read_csv(path).values
    x = np.array([np.append(x[i], [1.]) for i in range(len(x))])
    print (len(x))
    return x

def readData(xpath = 'data/X_train', ypath = 'data/Y_train', test_path = 'data/X_test'):
    x = pd.read_csv(xpath).values
    y = pd.read_csv(ypath).values.flatten()
    testx = pd.read_csv(test_path).values
    x = mix(x)
    testx = mix(testx)
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis = 1), y, np.concatenate((testx, np.ones((testx.shape[0], 1))), axis = 1)

def mix(x):
    #marital status
    z = np.concatenate((x[:, :32], (x[:, 32] + x[:, 33] + x[:, 34]).reshape(-1, 1)), axis = 1)

    #relationship
    z = np.concatenate((z, x[:, 35:53], x[:, 54:58], (x[:, 53] + x[:, 58]).reshape(-1, 1)), axis = 1)
    z = np.concatenate((z, x[:, 59:64]), axis = 1)
    x1 = x[:, 64] + x[:, 88] + x[:, 93] + x[:, 100] + x[:, 103]
    x2 = x[:, 65] + np.sum(x[:, 72:76], axis = 1) + x[:, 78] + x[:, 81] + x[:, 84] + x[:, 85] + x[:, 94] + x[:, 95] + x[:, 97] + x[:, 102] + x[:, 104]
    x3 = x[:, 66] + x[:, 80] + x[:, 87] + x[:, 91] + x[:, 99]
    x4 = np.sum(x[:, 67:72], axis = 1) + x[:, 76] + x[:, 77] + x[:, 79] + x[:, 86] + x[:, 89] + x[:, 90] + x[:, 92] + x[:, 96] + x[:, 101]
    x5 = x[:, 98] + x[:, 105]
    return np.concatenate((z, x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), x4.reshape(-1, 1), x5.reshape(-1, 1)), axis = 1)

def adjust(x):
    x1 = pd.get_dummies(x[:,1]//100000)
    return np.concatenate((x[:, 0:1], x1.values, x[:, 1:]), axis = 1)

def write_result(y, filenamey = 'k.csv'):
    with open(filenamey, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write(str(i + 1) + ',' + str(to_01(y[i][0])) + '\n')

def normalization(a):
    for i in [0, 1, 3, 4, 5]:
        mu_a = np.mean(a[:, i], axis = 0)
        sig_a = np.std(a[:, i], axis = 0)
        a[:, i] = (a[:, i] - mu_a)/sig_a

def to_01(y):
    return (y >= 0.5).astype(int)

def k():
    x, y = readTrainData()
    testx = readTestData()
    #x, y, testx = readData()
    print (x.shape, y.shape, testx.shape)
    normalization(x)
    normalization(testx)

    #x = np.concatenate((x[:, :59], x[:, 64:]), axis = 1)
    #testx = np.concatenate((testx[:, :59], testx[:, 64:]), axis = 1)
    #x = x[:, :64]
    #testx = testx[:, :64]

    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout
    from keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(Dense(1024, input_dim = x.shape[1], activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    #model.add(Dense(256, activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    #model.add(Dense(64, activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    #model.add(Dense(32, activation = 'sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    callbacks = [] 
    # callbacks.append(EarlyStopping(monitor='val_fmeasure', patience=25, verbose=1, mode='max'))
    callbacks.append(ModelCheckpoint('models/kkk_adam_aa.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
    split1, split2 = 10853, 21706
    #x, y, valid_x, valid_y = x[split1:], y[split1:], x[:split1], y[:split1]
    #x, y, valid_x, valid_y = np.concatenate((x[:split1], x[split2:]), axis =0), np.concatenate((y[:split1], y[split2:]), axis =0), x[split1:split2], y[split1:split2]
    x, y, valid_x, valid_y = x[:split2], y[:split2], x[split2:], y[split2:]
    model.fit(x, y, epochs = 250, 
        validation_data = (valid_x, valid_y), 
        shuffle = True, batch_size = 100, callbacks = callbacks)

    y = model.predict(testx)
    write_result(y, 'kkk.csv')

if __name__ == '__main__':
    k()