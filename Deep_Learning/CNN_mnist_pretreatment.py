# Author: RenFun
# File: CNN_mnist_pretreatment.py
# Time: 2022/01/06


# 比较有无预处理对CNN模型性能的影响
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import random


# 添加随机数种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 设置固定生成随机数的种子，使得每次运行该.py文件时生成的随机数相同
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # True的话，每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True
    # 在 PyTorch 中对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个。
    # 这样在模型启动的时候，只要额外多花一点点预处理时间，就可以较大幅度地减少训练时间。
    torch.backends.cudnn.benchmark = True


seed_everything(1024)


# 转换函数：将数据转换到tensor类型，再归一化使其服从（0.5， 0.5）的正态分布
def get_trans():
    # torchvision.transforms:常用的图片变换，例如裁剪、旋转等
    # torchvision.transforms.Compose：主要作用是串联多个图片变换的操作
    trans = torchvision.transforms.Compose(
        [
            # 改变数据类型
            torchvision.transforms.ToTensor(),
            # 数据标准化
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    return trans

# 不进行预处理
def get_trans_none():
    trans = torchvision.transforms.Compose(
        [
            # 改变数据类型
            torchvision.transforms.ToTensor(),
            # 数据标准化
            # torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    return trans

# 加载数据集，并进行预处理
train_data = torchvision.datasets.MNIST(
    root="./mnist",
    # 是否是训练集
    train=True,
    # 调用get_trans()，对数据进行转换
    transform=get_trans(),
    download=True
)
test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=get_trans(),
    download=True
)


# 加载未进行预处理的数据
train_data_none = torchvision.datasets.MNIST(
    root="./mnist",
    # 是否是训练集
    train=True,
    # 调用get_trans()，对数据进行转换
    transform=get_trans_none(),
    download=True
)
test_data_none = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=get_trans_none(),
    download=True
)

# 得到测试集的规模
def get_test_data_len():
    return len(test_data)


# 以BATCH_SIZE大小导入数据
def get_trainloader(BATCH_SIZE):
    # batch_size：一次训练所选取的样本数；shuffle：一代训练（epoch）中打乱数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def get_testloader(BATCH_SIZE):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def get_trainloader_none(BATCH_SIZE):
    # batch_size：一次训练所选取的样本数；shuffle：一代训练（epoch）中打乱数据
    train_loader_none = DataLoader(train_data_none, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader_none


def get_testloader_none(BATCH_SIZE):
    test_loader_none = DataLoader(test_data_none, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader_none


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MNIST数据集每张图片有28*28个像素，每个像素点用一个灰度值表示，输入规模1*28*28，即通道数*高度*宽度
# 训练集有60000张图片，测试集有10000张图片
# 搭建卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：提取特征信息
        # Sequential 这个表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.conv1 = nn.Sequential(
            nn.Conv2d(                  # 对输入数据进行二维卷积
                in_channels=1,          # 输入数据的通道数，例如RGB图片通道数为3，灰度图通道为1
                out_channels=16,        # 通道数，即卷积核数量
                kernel_size=5,          # 卷积核大小，此处为5*5，（m,n）表示m*n的卷积核，如果高宽相同，可以使用一个数字代替
                stride=1,               # 步长：每次卷积核滑动的行数或者列数
                padding=2),             # 选择填充，数值=(卷积核-1)/2，前提是stride=1；选择不填充，则padding=0
            nn.ReLU()                   # 激活函数为线性整流函数（修正线性单元）,相当于归一化
        )
        # 经过卷积层1，得到16*28*28的特征图（feature map），28 =（28+4-5）/1+1 即 （维数 + 2*padding - kernel_size）/stride +1。若不整除，则向下取整
        # 池化层1：提取重要特征信息，同时去掉不重要的信息，减少计算开销
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,          # 使用最大池化，池化核为2*2
                stride=2)               # stride的默认值为池化核的值
        )
        # 经过池化层1，输出的规模为16*14*14，（28-2）/2 +1，即w = (w - kernel_size)/stride +1。若不整除，则结果向上取整
        # 卷积层2：上一层的输出等于下一层的输入，设置输出通道数为32，卷积核5*5，stride=1，padding=2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU()
        )
        # 经过卷积操作后，得到32*14*14的特征图，（14+4-5）/1+1=14
        # 池化层2：池化核2*2，步长为2
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        # 经过池化操作，输出的规模为32*7*7，（14-2）/2+1=7
        # 全连接层：上层的输出是本层的输入，即输入数据为32*7*7，设置输出规模为10维，表示分类有10标签，分别对应1-10个数字
        self.full_connected = nn.Linear(in_features=32*7*7, out_features=10)            # 对输入数据进行线性转换
        # 全连接层是分类器角色，将特征映射到样本标记空间，本质是矩阵变换
    # 定义向前传播过程，过程名字不可更改，因为这是重写父类的方法
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # 将经过两次卷积池化操作后的输出拉伸为一行，输出数据规模为32*7*7
        x = x.view(x.size(0), -1)               # x,size(0)是batch_size,参数 -1 表示自适应
        output = self.full_connected(x)
        return output


# 定义常量
EPOCH = 20                              # 总的训练次数，即迭代次数
BATCH_SIZE = 128                        # 一批数据的规模，即一次训练选取的样本数量
LR = 0.01                                # 学习率
DOWNLOAD_MNIST = False                  # 运行代码时不需要下载数据集
# 训练误差
train_loss = []
# 精度
accuracy = []
# iteration次数
iteration = 0
# 实例化卷积神经网络
cnn1 = CNN()
cnn2 = CNN()
cnn1.to(DEVICE)
cnn2.to(DEVICE)
# torch.optim.选择优化函数，例如Adam，SGD ，AdaGrad ，RMSProp等
optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=LR)
optimizer2 = torch.optim.SGD(cnn2.parameters(), lr=LR)
# 损失函数选择交叉熵函数
loss_function = nn.CrossEntropyLoss()
# 加载训练集和测试集
train_loader = get_trainloader(BATCH_SIZE)
test_loader = get_testloader(BATCH_SIZE)
train_loader_none = get_trainloader_none(BATCH_SIZE)
test_loader_none = get_testloader_none(BATCH_SIZE)
# 对有预处理的模型进行训练
for i in range(EPOCH):
    x = 0
    print('epoch:', i+1)
    # 训练过程
    for data in train_loader:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        out = cnn1(img)
        # 得到训练误差
        loss = loss_function(out, label)
        x += loss.data.item()
        # train_loss.append(loss.data.item())
        # print('训练损失值：', loss.data.item())
        # 梯度归零
        optimizer1.zero_grad()
        # 反向传播误差，但是参数还没更新
        loss.backward()                           # 如果出现数据类型错误，将loss.backward(torch.ones_like(loss))
        # 更新模型参数
        optimizer1.step()
    train_loss.append(x/BATCH_SIZE)
    print('训练损失值：', x/BATCH_SIZE)
    num_correct = 0
    for data in test_loader:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        # 获得输出
        out = cnn1(img)
        # 获得测试误差
        loss = loss_function(out, label)
        # 将tensor类型的loss2中的data取出，添加到列表中
        # test_loss.append(loss.data.item())
        _, prediction = torch.max(out, 1)
        # 预测正确的样本数量
        num_correct += (prediction == label).sum()
        # 精度=预测正确的样本数量/测试集样本数量
    acc = float(format(num_correct.cpu().numpy() / float(get_test_data_len()), '0.4f'))  # .cpu()是将参数迁移到cpu上来
    # acc = float((prediction == label.data.cpu().numpy()).astype(int).sum()) / float(get_test_data_len.size(0))
    accuracy.append(acc)
    print('测试精度：', acc)
# 对无预处理的模型进行训练
for i in range(EPOCH):
    y = 0
    print('epoch:', i+1)
    # 训练过程
    for data in train_loader_none:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        out = cnn2(img)
        # 得到训练误差
        loss = loss_function(out, label)
        y += loss.data.item()
        # train_loss.append(loss.data.item())
        # print('训练损失值：', loss.data.item())
        # 梯度归零
        optimizer2.zero_grad()
        # 反向传播误差，但是参数还没更新
        loss.backward()                           # 如果出现数据类型错误，将loss.backward(torch.ones_like(loss))
        # 更新模型参数
        optimizer2.step()
    train_loss.append(y/BATCH_SIZE)
    print('训练损失值：', y/BATCH_SIZE)
    num_correct = 0
    for data in test_loader:
        img, label = data
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        # 获得输出
        out = cnn2(img)
        # 获得测试误差
        loss = loss_function(out, label)
        # 将tensor类型的loss2中的data取出，添加到列表中
        # test_loss.append(loss.data.item())
        _, prediction = torch.max(out, 1)
        # 预测正确的样本数量
        num_correct += (prediction == label).sum()
        # 精度=预测正确的样本数量/测试集样本数量
    acc = float(format(num_correct.cpu().numpy() / float(get_test_data_len()), '0.4f'))  # .cpu()是将参数迁移到cpu上来
    # acc = float((prediction == label.data.cpu().numpy()).astype(int).sum()) / float(get_test_data_len.size(0))
    accuracy.append(acc)
    print('测试精度：', acc)

# 绘制图像1：
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.plot(np.arange(1, 21), train_loss[20: 40], color='grey', label='无预处理')
plt.plot(np.arange(1, 21), train_loss[0: 20], color='lightblue', label='有预处理')
plt.xlabel("迭代次数")
plt.ylabel("训练损失值")
plt.xticks(np.arange(1, 21, 1))
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper right')
# plt.savefig('CNN_mnist_pretreatment.svg', bbox_inches='tight')
# plt.show()
plt.subplot(122)
plt.plot(np.arange(1, 21), accuracy[20: 40], color='grey', label='无预处理')
plt.plot(np.arange(1, 21), accuracy[0: 20], color='lightblue', label='有预处理')
plt.xlabel("迭代次数")
plt.ylabel("精度")
plt.xticks(np.arange(1, 21, 1))
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper left')
plt.savefig('CNN_mnist_pretreatment_loss&accuracy.svg', bbox_inches='tight')
plt.show()