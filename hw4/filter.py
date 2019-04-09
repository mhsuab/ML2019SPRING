import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as k

np.random.seed(3)

def g(n, i, iter_f):
    filter_images = []
    step = 1e-2
    for j in range(n):
        loss, grad = iter_f([i, 0])
        i += grad * step
        if j % 10 == 0:
            filter_images.append((i, loss))
    return filter_images

def depross_img(x):
    x = (x - np.mean(x))/(np.std(x) + 1e-30)
    x *= 0.1
    x += 0.5

    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

x = np.load('xvalid/try18.npy')
y = np.load('18y.npy')
p = np.load('18p.npy')
model = load_model('models/try18.h5')

layers = dict([layer.name, layer] for layer in model.layers)
input_img = model.input

collect_layers = [k.function([input_img, k.learning_phase()], [layers['conv2d_3'].output])]


for cnt, fn in enumerate(collect_layers):
    im = fn([x[100].reshape(1, 48, 48, 1), 0])
    fig = plt.figure(figsize = (14, 8))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        q = fig.add_subplot(nb_filter/16, 16, i + 1)
        q.imshow(im[0][0, :, :, i], cmap = 'Oranges')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.savefig('fig2_out.png')

collect_layers = [layers['conv2d_3'].output]
for cnt, c in enumerate(collect_layers):
    filter_img = []
    for i in range(64):
        input_img_data = np.random.random((1, 48, 48, 1))
        target = k.mean(c[:, :, :, i])
        grads = k.gradients(target, input_img)[0]
        grads /= (k.sqrt(k.mean(k.square(grads))) + 1e-10)
        fn = k.function([input_img, k.learning_phase()], [target, grads])

        filter_img.append(g(100, input_img_data, fn))
    print ('Finish Gradient')
    
    for it in range(100//10):
        fig = plt.figure(figsize = (14, 8))
        for i in range(64):
            q = fig.add_subplot(64/16, 16, i + 1)
            raw = filter_img[i][it][0].squeeze()
            q.imshow(depross_img(raw), cmap = 'Oranges')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            #plt.xlabel('{:.3f}'.format(filter_img[i][it][1]))
            plt.tight_layout()
        fig.savefig('fig2.png')
