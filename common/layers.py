import numpy as np
from functions import *


# ReLU
class ReLU:
    def __init__(self):
        # 内部属性，记录 x <= 0
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


# Sigmoid
class Sigmoid:

    def __init__(self):
        # 定义内部属性，记录输出值y，用于反向传播时计算梯度
        self.y = None

    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * (1.0 - self.y) * self.y
        return dx


# Affine
class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X进行保存，方便反向传播时计算梯度
        self.X = None
        self.orignal_x_shape = None
        # 将权重和偏置参数的梯度保存为属性，方便梯度下降时进行更新
        self.dW = None
        self.db = None

    def forward(self, X):
        self.orignal_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(self.X, self.W) + self.b
        return y

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        dX = dX.reshape(*self.orignal_x_shape)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX
    
# 输出层
class SoftmaxWithLoss:

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        n = self.t.shape[0]
        # 如果是one-hot编码的标签，直接代入公式计算
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果是顺序编码的标签，需要找到分类号对应的值，然后相减
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx / n
