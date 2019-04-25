import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
from torch import clamp
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import resnet50

model = resnet50(pretrained = True)
model.eval()

criterion = nn.CrossEntropyLoss()

px1_mean = np.asarray([0.485, 0.456, 0.406])
px1_std = np.asarray([0.229, 0.224, 0.225])
px2_mean = np.asarray([-0.485/0.229, -0.456/0.224, -0.406/0.225])
px2_std = np.asarray([1/0.229, 1/0.224, 1/0.225])
trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = px1_mean, std = px1_std)])
inverse_transform = transform.Compose([transform.Normalize(mean = px2_mean, std = px2_std), 
                                        transform.ToPILImage()])
inv1 = transform.Compose([transform.Normalize(mean = px2_mean, std = px2_std)])
inv2 = transform.Compose([transform.ToPILImage()])
count = 0.0

labels = pd.read_csv('labels.csv', usecols = ['TrueLabel']).values
same = []

def fgsm(image):
    image.requires_grad = True
    zero_gradients(image)
    output = model(image)
    loss = criterion(output, torch.tensor(labels[i]))
    loss.backward()
    image = image + epi * image.grad.sign_()
    image = image.detach()
    return image

for i in range(200):
    image = Image.open(os.path.join(sys.argv[1], str(i).rjust(3, '0') + '.png'))
    #image = Image.open(sys.argv[1] + str(i).rjust(3, '0') + '.png')
    image = trans(image)
    image = image.unsqueeze(0)
    epi = 0.003
    epoch = 20
    origin = np.argmax(model(image).detach().numpy())
    for _ in range(epoch):
        image = fgsm(image)
    n = np.argmax(model(image).detach().numpy())

    if origin != n:
        count += 1
    else:
        same.append(i)

    image = inv1(image[0])
    image = clamp(image, 0.0, 1.0)
    image = inv2(image)
    #print (origin, n)
    image.save(os.path.join(sys.argv[2], str(i).rjust(3, '0') + '.png'))
    #image.save(sys.argv[2] + str(i).rjust(3, '0') + '.png')

#print ('accuracy')
#print (count/200.)
#print ('same')
#print (same)
