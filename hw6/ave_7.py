import numpy as np
import pandas as pd
import sys
import os

import jieba
import emoji
from gensim.models import Word2Vec

from multiprocessing import Pool

input_path = sys.argv[1]
vocab_size = 11695
max_length = 36

jieba.set_dictionary(sys.argv[2])

test_x = pd.read_csv(input_path, usecols = ['comment']).values

def processJieba(l):
    # input : ['']
    # output: []
    return list(map(emoji.demojize, jieba.lcut(str(l[0]))))

P = Pool()
test_x = P.map(processJieba, test_x)
P.close()
P.join()

embed_rnn = Word2Vec.load('word2vec.bin')
embed_dnn = Word2Vec.load('word2vec_dnn.bin')

def sen2BOW(sen):
    bow = np.zeros(vocab_size)
    for i in range(len(sen)):
        try:
            bow[word2inx(sen[i])] += 1
        except:
            pass
    return bow

def word2inx(word):
    return embed_dnn.wv.vocab[word].index

def sen2vec(sen):
    sen = sen[:max_length]
    sen = list(map(lambda w: embed_rnn.wv[w], sen))
    sen += [np.zeros(embed_rnn.wv.vector_size)] * (max_length - len(sen))
    return sen

P = Pool()
test_x_dnn = P.map(sen2BOW, test_x)
P.close()
P.join()

P = Pool()
test_x_rnn = P.map(sen2vec, test_x)
P.close()
P.join()

test_x_dnn = np.array(test_x_dnn)
test_x_rnn = np.array(test_x_rnn)

np.save('hw6_test_x_dnn', test_x_dnn)
np.save('hw6_test_x_rnn', test_x_rnn)



