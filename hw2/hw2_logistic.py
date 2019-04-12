import numpy as np
import pandas as pd
import sys

def readTrainData(xpath = 'data/X_train', ypath = 'data/Y_train'):
    x = pd.read_csv(xpath).values
    x = np.array([np.append(x[i], [1.]) for i in range(len(x))])
    y = pd.read_csv(ypath).values
    return x, y.flatten()

def readTestData(path = 'data/X_test'):
    x = pd.read_csv(path).values
    x = np.array([np.append(x[i], [1.]) for i in range(len(x))])
    return x

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

def loss(y_hat, y):
    return (-1) * np.sum(y * np.log(y_hat + 10 ** (-8)) + (1-y) * np.log(1 - y_hat + 10 ** (-8)))

def to_01(y):
    return (y >= 0.5).astype(int)

def accu(y, y_hat):
    return np.sum(y == y_hat)/len(y)

def write_result(y, filenamey):
    with open(filenamey, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write(str(i + 1) + ',' + str(to_01(y[i])) + '\n')

def normalization(a):
    for i in [0, 1, 3, 4, 5]:
        mu_a = np.mean(a[:, i], axis = 0)
        sig_a = np.std(a[:, i], axis = 0)
        a[:, i] = (a[:, i] - mu_a)/sig_a

def logistic(xfile, yfile, tfile, ansfile):
    xtrain, ytrain = readTrainData(xfile, yfile)
    xtest = readTestData(tfile)
    normalization(xtrain)
    normalization(xtest)

    w = np.load('models/logistic.npy')
    write_result(sigmoid(np.dot(xtest, w)), ansfile)

if __name__ == '__main__':
    logistic(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
