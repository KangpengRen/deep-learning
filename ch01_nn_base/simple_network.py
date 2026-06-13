"""
简单的神经网络
  输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
            |m_1|
    |x_1|             |n_1|     |y_1|       
            |m_2|
    |x_2|             |n_2|     |y_2|
            |m_3|
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from common.functions import sigmoid, identify

def init_network():
    network = {}
    # 第一层参数
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # 第二层参数
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # 第三层参数
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    """
    前向传播

    Args:
        network (_type_): 神经网络
        x (_type_): 输入张量

    Returns:
        _type_: 输出张量
    """
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identify(a3)

    return y

# 定义模型
network = init_network()
# 定义数据
x = np.array([1.0, 0.5])

# 进行前向传播
y = forward(network, x)
print(y)