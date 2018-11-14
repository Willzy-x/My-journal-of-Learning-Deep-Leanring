import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
import math
import numpy as np
from torchvision import datasets as ds
from torch.autograd import Variable
from torch.utils.data import DataLoader

# import argparse
from ResNet import ResNet18

# Whether to use GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 50
pre_epoch = 0
BATCH_SIZE = 4
lr = 1e-3

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
])

train_set = ds.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                       train=True, transform=transform_train, download=True)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

testset = tv.datasets.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                              train=False, download=True, transform=transform_train)

testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet18()
# print(net)
# define optimization function
criterion = nn.CrossEntropyLoss()
# define optimization method
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


for epoch in range(num_epoch):
    running_loss = 0
    total_images = 0
    correct_images = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)

        _, predicts = t.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_images += labels.size(0)
        correct_images += (predicts == labels).sum()
        loss_data = loss.data[0].item()
        running_loss += loss_data
        if i % 2000 == 1999:
            print('Epoch, Mini-batch: [%d, %5d], loss: %.6f Training accuracy: %.5f' %
            (epoch + 1, i + 1, running_loss/2000, 100*correct_images.item()/total_images))

            running_loss = 0
            total_images = 0
            correct_images = 0

print("Finished training")
total = 0
correct = 0
for data in testloader:
    inputs, label = data
    label = Variable(label)
    inputs = Variable(inputs)
    output = net(inputs)
    _, predict = t.max(output.data, 1)
    total += label.size(0)
    correct += (predict == label).sum()

print('Test Accuracy: %.6f' % (100 * correct / total))
