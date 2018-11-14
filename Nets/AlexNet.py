from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision.transforms import ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

show = ToPILImage()
# 将数据类型转换为Tensor类型
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转为Tensor变量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ]
)
# 导入CIFAR10数据集 并放入 train_set
train_set = ds.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                       train=True, transform=transform, download=True)

# 封装为dataloader
train_loader = t.utils.data.DataLoader(train_set,
                                       batch_size=4, shuffle=True, num_workers=0)

# 同样，测试集
testset = tv.datasets.CIFAR10(root='/Users/yucheng/Downloads/R_data',
                              train=False, download=True, transform=transform)

testloader = t.utils.data.DataLoader(testset,
                                     batch_size=4, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = train_set[100]  # e.g.

print(data.size())
# 搭建AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(24, 96, 3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(96, 192, 3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 96 * 4 * 4)  # 开始全连接层的计算
        x = self.classifier(x)
        return x


net = AlexNet(10)
learning_rate = 1e-3  # 学习率
momentum = 0.9
weight_decay = 1e-5
print("Training with learning rate = %f, momentum = %f, lambda = %f " % (learning_rate, momentum, weight_decay))

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# 是否采取 weight_decay

# print(data.size(2))


# 训练
num_epoch = 40
for epoch in range(num_epoch):
    running_loss = 0
    total_images = 0
    correct_images = 0
    if epoch == 30:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate / 10, momentum=momentum, weight_decay=weight_decay)
        print("learning rate decay to 1/10...")
    if epoch == 40:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate / 100, momentum=momentum, weight_decay=weight_decay)
        print("learning rate decay to 1/100...")
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        print(inputs.size())
        inputs = Variable(inputs)
        labels = Variable(labels)
        # 清零梯度
        optimizer.zero_grad()
        # 正向传播
        outputs = net(inputs)
        # print(outputs, labels.size(0))
        _, predicts = t.max(outputs.data, 1)
        # 计算loss
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算训练集的准确度
        total_images += labels.size(0)
        correct_images += (predicts == labels).sum()
        loss_data = loss.data[0].item()
        running_loss += loss_data
        if i % 2000 == 1999:  # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
            print('Epoch, Mini—batch: [%d, %5d] loss: %.6f  Training set accuracy: %.6f ' %
                  (epoch + 1, i + 1, running_loss / 2000, 100 * correct_images.item() / total_images))
            # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
            total_images = 0
            correct_images = 0

    # print("Epoch [%d/%d], Loss: %.4f " % (epoch+1, num_epoch, running_loss))

# 测试集
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predict = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
