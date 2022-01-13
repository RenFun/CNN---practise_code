# Author: RenFun
# File: CNN_mnist.py
# Time: 2021/12/07


# 使用PyTorch搭建卷积神经网络模型，实现手写数字识别
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




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


train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,                 # 是否是训练集
    transform=get_trans(),      # 调用get_trans()，对数据进行转换
    download=True
)
test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=get_trans(),
    download=True
)


def get_trainloader(BATCH_SIZE):
    # batch_size：一次训练所选取的样本数；shuffle：一代训练（epoch）中打乱数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def get_testloader(BATCH_SIZE):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


# 是否使用gpu
def get_cuda_available():
    available = torch.cuda.is_available()
    return available


# 得到测试集的规模
def get_test_data_len():
    return len(test_data)


# 搭建卷积神经网络模型
class CNN(nn.Module):
    #
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：MNIST数据集每张图片有28*28个像素，每个像素点用一个灰度值表示，输入规模1*28*28，即（通道数，输入高度，输入宽度）
        # Sequential 这个表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.conv1 = nn.Sequential(
            nn.Conv2d(              # 对输入数据进行二维卷积
                in_channels=1,      # 输入数据的通道数，例如RGB图片通道数为3，灰度图通道为1
                out_channels=16,    # 通道数，即卷积核数量
                kernel_size=5,      # 卷积核大小，此处为5*5，（m,n）表示m*n的卷积核，如果高宽相同，可以使用一个数字代替
                stride=1,           # 步长：每次卷积核滑动的行数或者列数
                padding=2           # 选择填充，数值=(卷积核-1)/2，前提是stride=1；选择不填充，则padding=0
            ),
            # 经过卷积操作后，得到16*28*28的特征图（feature map），
            # 28 =（28+4-5）/1+1 即 （维数 + 2*padding - kernel_size）/stride +1
            # 此处计算若不整除，则向下取整
            nn.ReLU()               # 激活函数为线性整流函数（修正线性单元）,相当于归一化
        )
        # 池化层1：提取重要特征信息，同时去掉不重要的信息，减少计算开销
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,      # 使用最大池化，池化核为2*2
                stride=2)           # stride的默认值为池化核的值
            # w = (w - kernel_size)/stride +1     注意计算若不整除，则结果向上取整，stride 池化核默认为1
            # 经过池化操作，输出的规模为16*14*14，（28-2）/2 +1
        )
        # 卷积层2：第一层输出规模为16*14*14，第二层的输入等于第一层的输出，设置输出通道数为32，卷积核5*5，stride=1，padding=2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            # 经过卷积操作后，得到32*14*14的特征图
            nn.ReLU()
        )
        # 池化层2
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2)
            # 经过池化操作，输出的规模为32*7*7
        )
        # 全连接层：上层的输出是本层的输入，即输入数据为32*7*7，设置输出规模为10维，表示分类有10标签，分别对应1-10个数字
        # 全连接层是分类器角色，将特征映射到样本标记空间，本质是矩阵变换
        self.full_connected = nn.Linear(in_features=32*7*7, out_features=10)            # 对输入数据进行线性转换

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
EPOCH = 10                              # 总的训练次数，即迭代次数
BATCH_SIZE = 128                         # 一批数据的规模，即一次训练选取的样本数量
LR = 0.01                                # 学习率
DOWNLOAD_MNIST = False                  # 运行代码时不需要下载数据集
# 实例化卷积神经网络
cnn = CNN()
# 如果gpu可用，则使用gpu进行运算
cuda_available = get_cuda_available()
if cuda_available==True:
    cnn.cuda()
# 选择torch.optim.Adam作为模型参数的和优化函数，此外还有 SGD ，AdaGrad ，RMSProp等
# 实验表明，优化函数为SGD的模型迭代10次后平均精度为0.9875，优化函数为Adam的模型迭代10次后平均精度为0.9469
optimizer1 = torch.optim.SGD(cnn.parameters(), lr=LR)
# optimizer2 = torch.optim.SGD(cnn.parameters(), lr=LR)
# 损失函数选择交叉熵函数
loss_function = nn.CrossEntropyLoss()
# 加载训练集和测试集
train_loader = get_trainloader(BATCH_SIZE)
test_loader = get_testloader(BATCH_SIZE)
# 进行训练和测试
acc = []
train_loss = []
test_loss = []
for i in range(4):
    # 获取不同学习率的模型
    optimizer_lr = torch.optim.SGD(cnn.parameters(), lr=LR/10**(i+1))
    for ep in range(EPOCH):
        startTick = time.perf_counter()
        # 训练过程
        for data in train_loader:
            img, label = data
            if cuda_available:
                img = img.cuda()
                label = label.cuda()
            out = cnn(img)
            # 得到误差
            loss = loss_function(out, label)
            train_loss.append(loss)
            # print("误差：", loss)
            # 梯度归零
            optimizer_lr.zero_grad()
            # optimizer2.zero_grad()
            # 反向传播误差，但是参数还没更新
            loss.backward()
            # 更新模型参数
            optimizer_lr.step()
            # optimizer2.step()
        # 测试过程
        num_correct = 0
        # 预测正确样例数量
        for data in test_loader:
            img, label = data
            if cuda_available:
                img = img.cuda()
                label = label.cuda()
            # 获得输出
            out = cnn(img)
            # 得到测试误差
            loss_t = loss_function(out, label)
            test_loss.append(loss_t.data.item())
            #
            _, prediction = torch.max(out, 1)
            # 预测正确的样本数量
            num_correct += (prediction == label).sum()
        # 精度=预测正确的样本数量/测试集样本数量
        accuracy = format(num_correct.cpu().numpy()/get_test_data_len(), '0.4f')
        acc.append(float(accuracy))
        timeSpan = time.perf_counter() - startTick
        # print("迭代次数：", ep+1, "精度：", accuracy, "耗时：", timeSpan)
        print("迭代次数：", ep + 1, "耗时：", timeSpan)
    # print(acc)
# 绘制图像1：不同学习率对模型性能的影响
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, 11, 1), test_loss[0: 10], color='red', label="学习率=0.1")
plt.plot(np.arange(1, 11, 1), test_loss[10: 20], color='green', label="学习率=0.01")
plt.plot(np.arange(1, 11, 1), test_loss[20: 30], color='blue', label="学习率=0.001")
plt.plot(np.arange(1, 11, 1), test_loss[30: 40], color='yellow', label="学习率=0.0001")
plt.xlabel("迭代次数")
plt.ylabel("测试损失值")
plt.xticks(np.arange(1, 11, 1))
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper right')
# plt.savefig('CNN_mnist_learning_rate.svg')
plt.show()
