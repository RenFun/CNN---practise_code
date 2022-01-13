# Author: RenFun
# File: Test.py
# Time: 2022/01/02


# 使用PyTorch搭建卷积神经网络模型，实现手写数字识别
import os
import random
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# 运行结果写入文件
def writefile():
    with open('D:\PycharmProjects\Deep_Learning\CNN_mnist_train_loss.txt', 'w') as f:
        f.write(str(train_loss))
        f.close()
    with open('D:\PycharmProjects\Deep_Learning\CNN_mnist_accuracy.txt', 'w') as f:
        f.write(str(accuracy))
        f.close()


# 从文件中读取结果
def readfile():
    with open('D:\PycharmProjects\Deep_Learning\CNN_mnist_train_loss.txt', 'r') as f:
        content1 = f.read()
        print(content1)
        f.close()
    with open('D:\PycharmProjects\Deep_Learning\CNN_mnist_accuracy.txt', 'r') as f:
        content2 = f.read()
        print(content2)
        f.close()


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


# 加载训练集
train_data = torchvision.datasets.MNIST(
    root="./mnist",
    # 是否是训练集
    train=True,
    # 调用get_trans()，对数据进行转换
    transform=get_trans(),
    download=True
)
# 加载测试集
test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=get_trans(),
    download=True
)


# 以BATCH_SIZE导入训练样本
def get_trainloader(BATCH_SIZE):
    # batch_size：一次训练所选取的样本数；shuffle：一代训练（epoch）中打乱数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


# 以BATCH_SIZE导入测试样本
def get_testloader(BATCH_SIZE):
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


# 得到测试集的规模
def get_test_data_len():
    return len(test_data)


# 可视化混淆矩阵
def plot_matrix(cm, classes, cmap=plt.cm.Blues):
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    ax.imshow(cm, origin='upper', cmap=cmap)
    # 坐标轴设置
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,)
    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    # 标注信息
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if float(cm[i, j] * 100) > 0:
                ax.text(j, i, format(float(cm[i, j]), '.2f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xticks(rotation=30)
    plt.savefig('CNN_mnist_confusion_matrix.svg', bbox_inches='tight')
    plt.show()


# 是否使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 固定随机数种子，即可以复现上一次的结果
seed_everything(1024)


# MNIST数据集每张图片有28*28个像素，每个像素点用一个灰度值表示，输入规模1*28*28，即通道数*高度*宽度，训练集有60000个样本，测试集有10000个样本
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
Classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    # 数据集的标签：0~9
EPOCH = 20                                  # 总的训练次数，即迭代次数
BATCH_SIZE = 128                            # 一批数据的规模，即一次训练选取的样本数量
LR = 0.01                                   # 学习率
DOWNLOAD_MNIST = False                      # 运行代码时不需要下载数据集
# 训练误差
Train_loss = []
# 测试精度
accuracy = []
# 实例化卷积神经网络
cnn = CNN()
cnn.to(DEVICE)
# torch.optim.选择优化函数，例如Adam，SGD ，AdaGrad ，RMSProp等
optimizer1 = torch.optim.SGD(cnn.parameters(), lr=LR)
# 损失函数选择交叉熵函数
loss_function = nn.CrossEntropyLoss()
# 加载训练集和测试集
train_loader = get_trainloader(BATCH_SIZE)
test_loader = get_testloader(BATCH_SIZE)
# 测试集标签
Test_label = []
# 预测标签
PRE = []
for i in range(EPOCH):
    x = 0
    print('epoch:', i+1)
    # 训练过程
    for data in train_loader:
        train_img, train_label = data
        train_img = train_img.to(DEVICE)
        train_label = train_label.to(DEVICE)
        train_out = cnn(train_img)
        # 得到训练误差
        train_loss = loss_function(train_out, train_label)
        x += train_loss.data.item()
        # train_loss.append(loss.data.item())
        # print('训练损失值：', loss.data.item())
        # 梯度归零
        optimizer1.zero_grad()
        # 反向传播误差，但是参数还没更新
        train_loss.backward()                           # 如果出现数据类型错误，将loss.backward(torch.ones_like(loss))
        # 更新模型参数
        optimizer1.step()
    Train_loss.append(x/BATCH_SIZE)
    print('训练损失值：', x/BATCH_SIZE)
    # 测试CNN模型
    num_correct = 0
    for data in test_loader:
        test_img, test_label = data
        test_img = test_img.to(DEVICE)
        test_label = test_label.to(DEVICE)
        # 获得输出
        test_out = cnn(test_img)
        # 获得测试误差
        test_loss = loss_function(test_out, test_label)
        # 将tensor类型的loss2中的data取出，添加到列表中
        # test_loss.append(loss.data.item())
        _, prediction = torch.max(test_out, 1)
        # 预测正确的样本数量
        num_correct += (prediction == test_label).sum()
        # print(test_label.cpu().numpy().tolist())
        # 将测试标签的数据类型由tensor转换成list，并构成测试集全部样本的标签。
        Test_label.extend(test_label.cpu().numpy().tolist())    # .extend将待添加的的列表中的每个元素取出再添加，.append将待添加的列表作为一个整体进行添加
        PRE.extend(prediction.cpu().numpy().tolist())
    # 精度=预测正确的样本数量/测试集样本数量
    acc = float(format(num_correct.cpu().numpy() / float(get_test_data_len()), '0.4f'))  # .cpu()是将参数迁移到cpu上来
    # acc = float((prediction == label.data.cpu().numpy()).astype(int).sum()) / float(get_test_data_len.size(0))
    accuracy.append(acc)
    print('测试精度：', acc)
    # 每进行一次测试，得到一次PRF值和混淆矩阵，取最后一次PRF值进行记录，最后一次的混淆矩阵将覆盖之前保存的文件
    print(precision_score(Test_label, PRE, average='macro'))
    print(recall_score(Test_label, PRE, average='macro'))
    print(f1_score(Test_label, PRE, average='macro'))
    confusion_mat = confusion_matrix(Test_label, PRE)
    plot_matrix(confusion_mat, Classes)
# 将运行结果写入文件
writefile()
# 绘制图像1：
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(np.arange(1, len(Train_loss)+1), Train_loss, color='royalblue', label='训练损失值')
plt.xlabel("迭代次数")
plt.ylabel("训练损失值")
plt.xticks(np.arange(1, 21, 1))
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper right')
plt.savefig('CNN_mnist_train_loss.svg', bbox_inches='tight')
plt.show()


# 绘制图像2：
plt.plot(np.arange(1, len(accuracy)+1), accuracy, color='royalblue', label='精度')
plt.xlabel("迭代次数")
plt.ylabel("精度")
plt.xticks(np.arange(1, 21, 1))
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper left')
plt.savefig('CNN_mnist_accuracy.svg', bbox_inches='tight')
plt.show()
