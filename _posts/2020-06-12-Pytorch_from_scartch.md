---
title:  "Design your First Custom Neural Network from scratch Using PyTorch"
date:   2020-06-12
layout: single
author_profile: true
comments: true
---

### Design your First Custom Neural Network from scratch Using PyTorch

*Okay, here’s an original one : 
So, they ran a deep neural network to predict the hottest technological trend of 2014. Surprisingly, it predicted the answer to be deep neural networks! People accused it was biased. NN coders assumed it was probably due to the large initial bias.*😅

To jump directly to [code](https://github.com/Prudhvi0001/MNIST/blob/master/MNIST_Pytorch.ipynb).

[Here is a link for more best resources to learn Data Science.](https://github.com/Prudhvi0001/Data-science-best-resources)

If you already used TenserFlow or keras to create Neural Network architecture this is gonna be a cake walk for you. Although TenserFlow is great for creating models it doesn't’t support GPU as efficiently as PyTorch.

> The main purpose of PyTorch is to replace numpy with tensors which support GPU computation and unlike keras it provides maximum flexibility to customize to Network architecture.

Let’s not waste time and get started.!

Install PyTorch: Copy the command in this link and run in your terminal based on your system requirements [Link](https://pytorch.org/get-started/locally/).

**Import Libraries:**

```python
import torch

import torch.nn as nn # Contains Required functions and layers

import torch.nn.functional as F # For neural network functions:

# For Open ML datasets available in PyTorch.
from torchvision import datasets, transforms

# Contains Optimization function available in PyTorch.
import torch.optim as optim
```

Let’s try to build a 4 layer network with hello world data set of machine learning. [MNIST](http://yann.lecun.com/exdb/mnist/)

Import Dataset: MNIST Hand Written Digits

The data consists of a series of images (containing hand-written numbers) that are of the size `28 X 28`. We will discuss the images shortly, but our plan is to load the data into batches of `64`***.\***

PyTorch datasets library has all the popular datasets required to get you started.([Available Datasets](https://pytorch.org/docs/stable/torchvision/datasets.html))

```python
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# trainloader is what holds the data loader object which takes care of shuffling the data and constructing the batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False) # No need to shuffle test data.
```

Transform function above normalizes data to mean =0.5 and SD = 0.5

Load both train and test with same batch size(It doesn’t need to be **64**). each iteration of data gives 64 images and their labels.

Basic steps of neural network contains :

> **Forward Pass** -> **Loss calculation** -> **Backward Pass to optimize weights**

**Forward Pass:**

```python
class CustomNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Define Layers:
        self.l1 = nn.Linear(784, 256) # layer 1
        self.l2 = nn.Linear(256, 128) # layer 2
        self.l3 = nn.Linear(128, 64) # layer 3
        self.l4 = nn.Linear(64, 10) # layer 4

        # Define Activation functions:
        self.sigmoid = nn.Sigmoid() 
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim = 1) 


    def forward(self, x):
        """
        Layers: 4
        Activation Functions:
        RELU for first two layers
        Sigmoid for third layer
        Log Softmax for last layer
        """
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        x = self.l4(x)
        x = self.softmax(x)

        return x

    
NN = CustomNeuralNetwork()  # Intialize you NN
```

The nn.Module allows to override and create your own network architectures. You can even explore further by creating your own weights, bias and backward pass etc.

**Loss Calculation:**

```python
criterion = nn.NLLLoss() # Initialize loss function
```

Here I am using Negative log likelihood Loss. As our Data is a multi-class problem with Log Soft-max activation. [ReadHere](https://pytorch.org/docs/stable/nn.html#non-linear-activations-other)

**Optimizer:**

```python
optimizer = optim.Adam(NN.parameters(), lr = 0.001)
```

Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. [ReadHere](https://pytorch.org/docs/stable/optim.html)

**Train The Model (Back Propagation to update Weights):**

```python
# No:of times to train data
epochs = 5
for e in range(epochs):
    for images, labels in trainloader:
        # Faltten the images 
        images = images.view(images.shape[0], -1)

        # set optimizer gradients to zero:
        optimizer.zero_grad()

        
        output = NN(images) # Intial output
        loss = criterion(output, labels) # Loss  Caluclation
        loss.backward() # Pass loss function gradients to pervious layers:
        optimizer.step() # Update Weights

    print(loss.item()) # print loss for each epoch
```

Sample Output:

0.047625500708818436
0.0713285580277443
0.018748214468359947
0.023736275732517242
0.032160669565200806

Observe that for each epoch The loss reduces (***Don’t forget to set optimizers to zero_grad() before initialization.\***)

Best Practice would be save your model for further use so that you need to train again: Using PyTorch you can save either model or state_dict() which requires less space.

```python
# Save your model
PATH = './NeuralNet.pth'
torch.save(NN.state_dict(), PATH) 
# Load your model 
NN = CustomNeuralNetwork()
NN.load_state_dict(torch.load(PATH))
```

**Predict on Test Data:**

```python
# Accuracy on Test Data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        output = NN(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

Sample Output:

Accuracy of the network on the 10000 test images: 97 %

**Accuracy of each label:**

```python
classes = ('0','1','2','3','4','5','6','7','8','9')
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(images.shape[0], -1)
        outputs = NN(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Accuracy of each class:
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

Sample Output:

```python
Accuracy of     0 : 100 %
Accuracy of     1 : 100 %
Accuracy of     2 : 92 %
Accuracy of     3 : 98 %
Accuracy of     4 : 98 %
Accuracy of     5 : 90 %
Accuracy of     6 : 96 %
Accuracy of     7 : 98 %
Accuracy of     8 : 98 %
Accuracy of     9 : 96 %
```

We can observe that the model is confusing for 2 and 5, A simple solution would be collect more data of 2’s and 5’s or you explore options like Data Augmentation, Shift images to center before passing it train your model.

For More Advanced Practice refer [Elvis](https://medium.com/dair-ai/pytorch-1-2-introduction-guide-f6fa9bb7597c).

If you are still struck at making a decision to learn tensorflow or pytorch like me at the start. [ReadThis](https://towardsdatascience.com/tensorflow-vs-pytorch-the-battle-continues-9dcd34bb47d4).