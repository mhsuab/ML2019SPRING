import numpy as np
import pandas as pd
from keras.models import load_model
import sys

def readTestData(path = 'data/X_test'):
    x = pd.read_csv(path).values
    x = np.array([np.append(x[i], [1.]) for i in range(len(x))])
    return x

def to_01(y, edge = 0.5):
    return (y >= edge).astype(int)

def normalization(a):
    for i in [0, 1, 3, 4, 5]:
        mu_a = np.mean(a[:, i], axis = 0)
        sig_a = np.std(a[:, i], axis = 0)
        a[:, i] = (a[:, i] - mu_a)/sig_a

def write_result(y, filenamey = 'k.csv', edge = 0.5):
    with open(filenamey, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write(str(i + 1) + ',' + str(to_01(y[i][0], edge)) + '\n')

model = load_model(sys.argv[1])
xtest = readTestData(sys.argv[2])
normalization(xtest)
ypre = model.predict(xtest)
write_result(ypre, sys.argv[3])