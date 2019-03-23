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

def train_log(x, y, lr = 0.1, epoch = 1000):
    w = np.zeros(x.shape[1])
    lr_w = np.zeros(w.shape)

    for i in range(epoch):
        y_hat = sigmoid(np.dot(x, w))
        w_grad = (2 * np.dot(np.transpose(x), (y_hat - y)))
        lr_w += w_grad ** 2
        w -= lr/np.sqrt(lr_w) * w_grad
        print ('epoch = {}, loss = {:7.5f}, accu = {:.7f}%'.format(i, loss(y_hat, y), 100 * accu(y, to_01(y_hat))))
    return w, lr_w

def sigmoid(z):
    return (1/(1 + np.exp(-z)))

def loss(y_hat, y):
    return (-1) * np.sum(y * np.log(y_hat + 10 ** (-8)) + (1-y) * np.log(1 - y_hat + 10 ** (-8)))

def to_01(y):
    return (y >= 0.5).astype(int)

def accu(y, y_hat):
    return np.sum(y == y_hat)/len(y)

def write_result(y, w, lr_w, filenamey, filenamew, filenamelr_w):
    with open(filenamey, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write(str(i + 1) + ',' + str(to_01(y[i])) + '\n')
    np.save(filenamew, w)
    np.save(filenamelr_w, lr_w)

def normalization(a, l):
    for i in l:
        mu_a = np.mean(a[:, i], axis = 0)
        sig_a = np.std(a[:, i], axis = 0)
        a[:, i] = (a[:, i] - mu_a)/sig_a

def logistic():
    xtrain, ytrain = readTrainData()
    xtest = readTestData()
    norm = [0, 1, 3, 4, 5]
    normalization(xtrain, norm)
    normalization(xtest, norm)
    print (xtrain.shape, ytrain.shape, xtest.shape)

    w, lr_w = train_log(xtrain, ytrain, 0.1, 10000)
    write_result(sigmoid(np.dot(xtest, w)), w, lr_w, 'lna_2_4.csv', 'models/lna_2_4', 'lr_w/lna_2_4')

if __name__ == '__main__':
    logistic()
