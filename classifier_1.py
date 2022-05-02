# Machine Learning: Artificial Neural Network (ANN)
# Date: 17 April 2022
# Developer: Theo Madikgetla

import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# Sets makes the sample image sizes to be fixed.

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                              ])

# Creates the training set and test set.

training_set = datasets.MNIST(root='input', download=False, train=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='input', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(training_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# Plot to view an image

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
plt.show()

figure = plt.Figure()
num_of_images = 60

for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

plt.show()

# Creating Neural Network model using the sequential model
# Three layer with different activation functions

model = nn.Sequential(
                        nn.Linear(784, 128),    # 1st layer
                        nn.ReLU(),              # Activation function: Rectified Linear Function

                        nn.Linear(128, 64),     # 2nd Layer
                        nn.Tanh(),              # Activation Function: Hyperbolic Tangent Function

                        nn.Linear(64, 10),      # 3rd Layer
                        nn.LogSoftmax(dim=1)    # Activation Function: Log Softmax
                     )

print(model)

















