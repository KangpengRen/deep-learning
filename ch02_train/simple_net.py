import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error

class SimpleNet:

    def __init__(self):
        # 权重初始化
        self.W = np.random.rand(2, 3)  # 2行3列的权重矩阵

    def forward(self, X):
        # 前向传播
        a = np.dot(X, self.W)  # 输入X与权重W的点积
        y = softmax(a)  # 通过softmax函数得到输出y
        return y
    
    def loss(self, X, t):
        y = self.forward(X)  # 计算前向传播的输出
        loss = cross_entropy_error(y, t)  # 计算损失
        return loss
    
if __name__ == '__main__':
    # 1. 定义数据
    X = np.array([0.6, 0.9])  # 输入数据
    t = np.array([0, 0, 1])  # 目标标签（one-hot编码）

    # 2. 创建网络实例
    net = SimpleNet()

    # 3. 计算梯度
    f = lambda _: net.loss(X, t)
    grad_w = numerical_gradient(f, net.W)

    print(grad_w)
