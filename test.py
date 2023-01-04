
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar



device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Building model..')

from model import Model
net=Model(num_classes=47)
net.eval()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

import glob
from PIL import Image
from torchvision import transforms as T

for input_image in glob.glob('*.jpg'):
    print(input_image)
    input_image=Image.open(input_image).convert('RGB')
    preprocess=T.Compose([
            transforms.Resize(size=(256,256)),
            # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    input_tensor=preprocess(input_image)
    input_batch=input_tensor.unsqueeze(0)
    with torch.no_grad():
        output=net(input_batch)
    # print(output.size())
    # print(model)
    prob=torch.nn.functional.softmax(output[0],dim=0)
    # print(prob)
    top5_prob,top_catid= torch.topk(prob,1)
    classes=['','none']
    for i in range(top5_prob.size()[0]):
        print(classes[top_catid[i]],top5_prob[i].item())
