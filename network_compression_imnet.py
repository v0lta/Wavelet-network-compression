import torch
import collections
import numpy as np
import torchvision
import pywt
import matplotlib.pyplot as plt
import torchvision.models as models
from wave.wavelet_linear import WaveletLayer
from fastfood.fastfood import FastFoodLayer
from torch.utils.tensorboard.writer import SummaryWriter

epochs = 2
batch_size = 512
learning_rate = 0.001
runs = 20
wavelet = False
fastfood = True
# test

normalize = torchvision.transforms.Compose(
    [torchvision.transforms.RandomResizedCrop(224),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])


image_net = torchvision.datasets.ImageNet(root='./data_sets/image_net/', transform=normalize)
image_net_test = torchvision.datasets.ImageNet(root='./data_sets/image_net/', transform=normalize, split='val')

image_net_loader = torch.utils.data.DataLoader(image_net, batch_size=batch_size, shuffle=True, num_workers=12,
                                               pin_memory=True)
test_image_net_loader = torch.utils.data.DataLoader(image_net_test, batch_size=batch_size, shuffle=False,
                                                    num_workers=12, pin_memory=True)
