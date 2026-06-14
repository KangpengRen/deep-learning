import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from common.functions import softmax, sigmoid, cross_entropy_error
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.params['b1'] = np.zeros(hidden_size)  # 隐藏层的偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重
        self.params['b2'] = np.zeros(output_size)  # 输出层的偏置

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a1 = np.dot(X, W1) + b1  # 输入层到隐藏层的线性变换
        z1 = sigmoid(a1)  # 隐藏层的激活函数
        a2 = np.dot(z1, W2) + b2  # 隐藏层到输出层的线性变换
        y = softmax(a2)  # 输出层的激活函数（softmax）

        return y
    
    def loss(self, X, t):
        y = self.forward(X)  # 计算预测值
        return cross_entropy_error(y, t)  # 计算损失
    
    def accuracy(self, X, t):
        y_proba = self.forward(X)  # 计算预测值
        y = np.argmax(y_proba, axis=1)  # 取预测值中概率最大的索引
        accuracy = np.sum(y == t) / float(X.shape[0])  # 计算准确率
        return accuracy
    
    def numerical_gradient(self, X, t):
        # 定义目标函数
        loss_f = lambda _: self.loss(X, t)  # 计算损失的函数
        # 对每个参数用数值微分方法计算梯度
        grads = {}
        grads['W1'] = numerical_gradient(loss_f, self.params['W1'])  # W1的梯度
        grads['b1'] = numerical_gradient(loss_f, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_f, self.params['W2'])  # W2的梯度
        grads['b2'] = numerical_gradient(loss_f, self.params['b2'])
        return grads