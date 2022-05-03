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
                        nn.ReLU(),              # Activation Function: Hyperbolic Tangent Function

                        nn.Linear(64, 10),      # 3rd Layer
                        nn.Softmax(dim=1)       # Activation Function: Softmax Function
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
print("\n")
print("Now training Model.....")
print("----------------------------------------------")
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

def classify(img, ps):
    '''
    Function for viewing an image and it's predicated classes. They will be shown to you.

    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

images, labels = next(iter(testloader))
# replace training_loader to check training accuracy.

img = images[0].view(1, 784)

# Turns off gradients to speed up this part
with torch.no_grad():
    log_prob = model(img)

# Output of the network are log_probabilities, need to take exponential for probabilities
pb = torch.exp(log_prob)
probab = list(pb.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
classify(img.view(1, 28, 28), pb)

# This part of the code validates the model
correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        with torch.no_grad():
            logps = model(img)
    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if (true_label == pred_label):
        correct_count += 1
    all_count += 1

print("======================================================")
print("Number of Images Tested =", all_count)
print("\nModel Accuracy = ", (correct_count/all_count))
print("======================================================")
# This line of code saves the model then prints "DONE".
print("Saving model.....")
torch.save(model, 'output/A3_Model_5.pt')
print("Done")

print("***************** [USER INTERACTION] *****************")
user_input = ''
while user_input != 'exit':
    print("Please enter a filepath: ")
    user_input  = input("> ")
    if user_input != 'exit':
        ''' Loads up our saved model.'''
        net = torch.load("output/A3_Model_5.pt")

        ''' Loads up the image on the path entered by user'''
        image_ = Image.open(user_input)
        img_ = TF.to_tensor(image_).unsqueeze(0)
        img__ = img_.view(img_.shape[0], -1)
        with torch.no_grad():
            log_prob_ = model(img__)
        pb_ = torch.exp(log_prob_)
        probab_ = list(pb_.numpy()[0])
        print("Classifier:", probab_.index(max(probab_)))
        classify(img__.view(1, 28, 28), pb)

    else:
        print("Exiting....")

