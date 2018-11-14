from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import torchvision as tv
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
from torch import optim
from matplotlib import pyplot as plt

transform = ts.Compose(
    [
        ts.ToTensor(),
        ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = ds.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                       train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

testset = tv.datasets.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                              train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv1
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),  # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(32, 64, 3, padding=1),  # Conv3
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(64, 128, 3, padding=1),  # Conv5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv6
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv7
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool3
            nn.Conv2d(128, 256, 3, padding=1),  # Conv8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv9
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool4
            nn.Conv2d(256, 256, 3, padding=1),  # Conv11
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv12
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv13
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2)  # Pool5 是否还需要此池化层，每个通道的数据已经被降为1*1
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


net = VGGNet(10)
lr = 1e-3
momentum = 0.9

num_epoch = int(input("Please choose the number of the epochs: "))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
print('Training with learning rate = %f, momentum = %f ' % (lr, momentum))

loss_p = np.array([])
e = np.linspace(0, num_epoch-1, num_epoch)
for t in range(num_epoch):
    running_loss = 0
    running_loss_sum_per_epoch = 0
    total_images = 0
    correct_images = 0
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(images)
        _, predicts = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_images += labels.size(0)
        correct_images += (predicts == labels).sum().item()
        loss_data = loss.item()
        running_loss += loss_data

        running_loss_sum_per_epoch = loss.detach().numpy() + running_loss_sum_per_epoch
        if i % 2000 == 1999:
            print('Epoch, batch [%d, %5d] loss: %.6f, Training accuracy: %.5f' %
                  (t + 1, i + 1, running_loss / 2000, 100 * correct_images / total_images))
            running_loss = 0
            total_images = 0
            correct_images = 0

    loss_p = np.append(loss_p, running_loss_sum_per_epoch)

print('Finished training.')
plt.plot(e, loss_p, color='red', linestyle='--', labels='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
plt.show()
