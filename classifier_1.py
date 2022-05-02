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

# Defining the negative log-likelihood for calculating loss
criterion = nn.NLLLoss()

images, labels = next(iter(training_loader))
images = images.view(images.shape[0], -1)

# log probabilities 
logps = model(images)

# Calculate the NLL-Loss
loss = criterion(logps, labels)

print('Before backward pass: \n', model[0].weight.grad)
# Calculates the gradients of parameter
loss.backward()
print('After backward: \n', model[0].weight.grad)
print('\n')

'''The following code trains the neural network '''
# defining the optimiser with stochastic gradient descent and default parameter

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print('Initial weights - ', model[0].weight)
print('\n')

images, labels = next(iter(training_loader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward Pass
output = model(images)
loss = criterion(output, labels)

# The backward pass and the update weights
loss.backward()
print('Gradient - ', model[0].weight.grad)

time_ = time()
epochs = 15        # Total number of iterations(epochs) for training

running_lost_list = []
epochs_list = []

for eps in range(epochs):
    running_loss = 0
    for images, labels in training_loader:
        # Flatenning MNIST images with size [64, 784]
        images = images.view(images.shape[0], -1)

        # Defining gradient in each epoch as 0
        optimizer.zero_grad()

        # Modeling for each image batch
        output = model(images)

        # Calculating the loss
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        # Calculate the loss
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}". format(eps, running_loss/len(training_loader)))
print("\nTraining Time (in minutes) = ", (time() - time_)/60)
















