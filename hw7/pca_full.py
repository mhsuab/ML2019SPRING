import os
import sys
import numpy as np
from skimage.io import imread, imsave

run_type = sys.argv[1]
IMAGE_PATH = sys.argv[2]
if run_type == 'f':
    input_img = sys.argv[3]
    recon_img = sys.argv[4]

k = 5

def process(k):
    M = np.copy(k)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

filelist = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.jpg')]

img_shape = (600, 600, 3)

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH, filename))
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')

mean = np.mean(training_data, axis = 0)
training_data -= mean

u, s, v = np.linalg.svd(training_data.T, full_matrices = False)

def reconstructNsave(originfilename, reconstuctfilename):
    x = imread(originfilename).flatten().astype('float32')
    x -= mean
    weight = np.dot(x, u[:, :k])
    imsave(reconstuctfilename, (process(mean + np.dot(weight, u[:, :k].T))).reshape(img_shape))

if run_type == 'r':
    #1.c
    test_image = ['1.jpg', '10.jpg', '22.jpg', '37.jpg', '72.jpg']
    for x in test_image:
        reconstructNsave(os.path.join(IMAGE_PATH, x), 'report2/' + x[:-4] + '_reconstruct.jpg')

    #1.a
    imsave('report2/average.jpg', (process(mean)).reshape(img_shape))

    #1.b
    for i in range(5):
        imsave('report2/' + str(i) + '_eigenface.jpg', (u[:, i]).reshape(img_shape))

    #1.d
    for i in range(5):
        print (s[i] * 100 / sum(s))

elif run_type == 'f':
    reconstructNsave(os.path.join(IMAGE_PATH, input_img), recon_img)
