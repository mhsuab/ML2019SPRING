import numpy as np
import pandas as pd
import sys
import os
from keras.models import load_model

from multiprocessing import Pool

files = ['stack_2.h5', 'stack_4.h5', 'stack_5.h5', 'stack_7.h5', 'stack_8.h5', 'stack_3.h5', 'dnn.h5']

output_path = sys.argv[1]

test_x_dnn = np.load('hw6_test_x_dnn.npy')
test_x_rnn = np.load('hw6_test_x_rnn.npy')

model = load_model(os.path.join('models', files[-1]))
y = model.predict(test_x_dnn)

for s in files[:-1]:
    model = load_model(os.path.join('models', s))
    y += model.predict(test_x_rnn)


ans = ((np.array(y)/7.) >= 0.5).astype(int)

with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i in range(len(ans)):
        f.write(str(i) + ',' + str(ans[i][0]) + '\n')






