from keras.models import load_model
from sklearn.cluster import KMeans

import sys
import numpy as np
import pandas as pd
import read as r
from sklearn.decomposition import PCA

n_components = int(sys.argv[1])
img_path = sys.argv[2]
test_case_path = sys.argv[3]
prediction_filename = sys.argv[4]
model_path = sys.argv[5]

img = r.read(img_path, False)
img = (img.astype('float32')/255.)
encoder = load_model(model_path)
img = encoder.predict(img)

img = img.reshape(img.shape[0], -1)
img = PCA(n_components = n_components, whiten = True, random_state = 1346).fit_transform(img)

cluster = KMeans(n_clusters = 2, random_state = 1346).fit(img)

files = pd.read_csv(test_case_path).values

with open(prediction_filename, 'w') as f:
    f.write('id,label\n')
    for (i, j, k) in files:
        p1 = cluster.labels_[j - 1]
        p2 = cluster.labels_[k - 1]
        f.write('{},{}\n'.format(i, 1 if p1 == p2 else 0))

