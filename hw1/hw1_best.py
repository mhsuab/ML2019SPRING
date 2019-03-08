import numpy as np
import pandas as pd
import sys

def read_train(filename = 'train.csv'):
    raw = pd.read_csv(filename, encoding = 'Big5').values[:, 3:]
    for i in raw:
        for j in range(len(i)):
            if i[j] == 'NR':
                i[j] = '0.0'
    raw = raw.astype('float')
    X, Y = [], []
    for i in range(0, raw.shape[0], 18*20):
        concat = np.concatenate(np.vsplit(raw[i: i + 18 * 20], 20), axis = 1)
        for j in range(concat.shape[1] - 9):
            X.append(np.append(concat[:, j:j+9].reshape(-1), [1.]))
            Y.append(concat[9][j + 9])
    return np.array(X), np.array(Y)

def preprocess(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] < 0:
                if j == 0:
                    prep_start(x, i)
                elif j == (len(x[0])) - 1:
                    prep_end(x, i)
                else:
                    prep(x, i, j)

def prep_start(x, i):
    j = 1
    while j < len(x[i]):
        if x[i][j] > 0:
            break
        j += 1
    delta = x[i][j+1] - x[i][j]
    while j > 0:
        x[i][j - 1] = (x[i][j] - delta) if (x[i][j] - delta) > 0 else 0.0
        j -= 1


def prep_end(x, i):
    j = -2
    while j > ((-1) * len(x[i]) + 1):
        if x[i][j] > 0:
            break
        j -= 1
    delta = x[i][j - 1] - x[i][j]
    while j < -1:
        x[i][j + 1] = (x[i][j] - delta) if (x[i][j] - delta) > 0 else 0.0
        j += 1

def prep(x, i, j):
    for k in range(j, len(x[i])):
        if x[i][k] > 0:
            break
    delta = (x[i][k] - x[i][j - 1])/(k - j + 1)
    while k > j:
        x[i][k - 1] = (x[i][k] - delta) if (x[i][k] - delta) > 0 else 0.0
        k -= 1


def read_test(filename = 'test.csv'):
    raw = pd.read_csv(filename, encoding = 'Big5', header = None).values[:, 2:]
    for i in raw:
        for j in range(len(i)):
            if i[j] == 'NR':
                i[j] = '0.0'
    raw = raw.astype('float')
    X = []
    for i in np.vsplit(raw, raw.shape[0] // 18):
        X.append(np.append(i.reshape(-1), [1.]))
    return np.array(X)

def train(x, y, batch_size = 471):
    from sklearn.utils import shuffle
    w = np.ones(x.shape[1])
    lr_w = np.ones(w.shape)
    lr = 1
    epoch = 300000
    print (x.shape, y.shape)
    n = x.shape[0]//batch_size
    m = 0

    for i in range(epoch):
        pre = np.dot(x[m:m+batch_size], w) - y[m:m+batch_size]
        print ('epoch={}, loss={}'.format(i, sum(pre ** 2)/len(y[m:m+batch_size])))
        w_grad = 2 * np.dot((x[m:m+batch_size]).T, pre)
        lr_w += w_grad ** 2
        w -= lr/np.sqrt(lr_w) * w_grad
        m += 1
        if m == n:
            x, y = shuffle(x, y)
            m = 0
    return w, lr_w

def write_result(y, w, lr_w, filenamey = 'wb.csv', filenamew = 'models/wb.txt', filenamelr_w = 'lr_w/wb.txt'):
    #write y1
    with open(filenamey, 'w') as f:
        f.write('id,value\n')
        for i in range(len(y)):
            if y[i] > 0:
                f.write('id_' + str(i) + ',' + str(y[i]) + '\n')
            else:
                f.write('id_' + str(i) + ',' + str(0.0) + '\n')
    np.savetxt(filenamew, w, delimiter='\n')
    np.savetxt(filenamelr_w, lr_w, delimiter='\n')

def load():
    w = np.loadtxt('models/wb.txt', delimiter = '\n')
    x = read_test(sys.argv[1])
    y = np.dot(x, w)
    with open(sys.argv[2], 'w') as f:
        f.write('id,value\n')
        for i in range(len(y)):
            if y[i] > 0:
                f.write('id_' + str(i) + ',' + str(y[i]) + '\n')
            else:
                f.write('id_' + str(i) + ',' + str(0.0) + '\n')

def main():
    trainx, trainy = read_train()
    preprocess(trainx)
    testx = read_test()
    print (trainx.shape, trainy.shape, testx.shape)
    w, lr_w = train(trainx, trainy)
    ans = []
    for arr in testx:
        ans.append(np.dot(arr, w))
    write_result(np.dot(testx, w), w, lr_w)
    print (ans[:10])
    print (np.dot(testx, w)[:10])

if __name__ == '__main__':
    load()



