# Author: RenFun
# File: Tool.py
# Time: 2022/01/10


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.arange(-10, 10, 0.1)
y = np.maximum(0, x)
plt.plot(x, y, label='ReLU函数')
plt.grid(b=True, linestyle='--')
plt.legend(loc='upper left')
plt.savefig('ReLU.svg', bbox_inches='tight')
plt.show()