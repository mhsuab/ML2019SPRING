import sys
import numpy as np
import pandas as pd
from keras.models import load_model

test = np.array([i.split(' ') for i in pd.read_csv(sys.argv[1])['feature']]).astype('float')
test = (test/255.).reshape((-1, 48, 48, 1))

modelname = ['models/try10.h5', 'models/try11.h5', 'models/try13.h5', 'models/try15.h5', 'models/try12.h5']

model = load_model(modelname[0])
predict = model.predict(test)
ans = (predict.argmax(axis=1)[:,None] == range(predict.shape[1])).astype(int)
ans *= 2

for i in modelname[1:]:
	model = load_model(i)
	predict = model.predict(test)
	ans += (predict.argmax(axis=1)[:,None] == range(predict.shape[1])).astype(int)

ans = np.argmax(ans, axis = 1)

with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    for i in range(len(ans)):
        f.write(str(i) + ',' + str(ans[i]) + '\n')