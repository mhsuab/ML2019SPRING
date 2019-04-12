import numpy as np
import pandas as pd
from lime import lime_image
from skimage.segmentation import slic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as k
import sys

np.random.seed(5)
model = load_model('models/try18.h5')

path = sys.argv[2] + '/'
raw = pd.read_csv(sys.argv[1], nrows = 300)
x = np.array([i.split(' ') for i in raw['feature']]).astype('float')
y = pd.get_dummies(raw['label']).values
x = (x/255.).reshape((-1, 48, 48, 1))
y = np.argmax(y, axis = 1)

#lime
d = {0:10, 1:299, 2:5, 3:7, 4:3, 5:15, 6:4}

def predict(data):
    pred = model.predict(data[:, :,:, 0].reshape(-1, 48, 48, 1))
    return pred

def segmentation(data):
    return slic(data)

def x2rgb(x):
    tmp = np.concatenate((x, x, x), axis = 2)
    return tmp

for i in range(7):
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image = x2rgb(x[d[i]]), classifier_fn = predict, segmentation_fn = segmentation)
    image, mask = explaination.get_image_and_mask(label = y[d[i]], positive_only = False, hide_rest = False, num_features = 5, min_weight = 0.0)
    plt.imsave((path + 'fig3_{}.jpg').format(i), image)
print ('finish lime')


#filter
def g(n, i, iter_f):
    filter_images = []
    for j in range(n):
        loss, grad = iter_f([i, 0])
        i += grad * 0.01
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

layers = dict([layer.name, layer] for layer in model.layers)
input_img = model.input

collect_layers_o = [k.function([input_img, k.learning_phase()], [layers['conv2d_3'].output])]
collect_layers_i = [layers['conv2d_3'].output]


for cnt, fn in enumerate(collect_layers_o):
    im = fn([x[0].reshape(1, 48, 48, 1), 0])
    fig = plt.figure(figsize = (14, 8))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        q = fig.add_subplot(nb_filter/16, 16, i + 1)
        q.imshow(im[0][0, :, :, i], cmap = 'Oranges')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer conv2d_3 (Given image 0)')
    fig.savefig(path + 'fig2_2.jpg')

for cnt, c in enumerate(collect_layers_i):
    filter_img = []
    for i in range(nb_filter):
        input_img_data = np.random.random((1, 48, 48, 1))
        target = k.mean(c[:, :, :, i])
        grads = k.gradients(target, input_img)[0]
        grads /= (k.sqrt(k.mean(k.square(grads))) + 1e-10)
        fn = k.function([input_img, k.learning_phase()], [target, grads])
        filter_img.append(g(100, input_img_data, fn))
    
    for it in range(10):
        fig = plt.figure(figsize = (14, 8))
        for i in range(nb_filter):
            q = fig.add_subplot(nb_filter/16, 16, i + 1)
            raw = filter_img[i][it][0].squeeze()
            q.imshow(depross_img(raw), cmap = 'Oranges')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Filters of layer conv2d_3')
        fig.savefig(path + 'fig2_1.jpg')
print ('finish filter')

#saliency
for i in range(7):
    val = model.predict(x[d[i]].reshape(-1, 48, 48, 1))
    pred = val.argmax(axis = -1)
    target = k.mean(model.output[:, y[d[i]]])
    grads = k.gradients(target, input_img)[0]
    fn = k.function([input_img, k.learning_phase()], [grads])

    #heatmap
    v = fn([x[d[i]].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
    v *= -1
    v = np.max(np.abs(v), axis = -1, keepdims = True)
    v = (v - np.mean(v))/(np.std(v) + 1e-10)
    v *= 0.1
    v /= np.max(v)
    heatmap = (v).reshape(48, 48)

    plt.figure()
    plt.imshow(heatmap, cmap = plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig((path + 'fig1_{}.jpg').format(i), dpi = 100)