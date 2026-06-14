import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient

# 定义梯度下降法的函数
def gradient_desent(f, init_x, lr=0.01, num_iter=100):
    x = init_x
    # 定义x历史列表
    x_history = []
    for _ in range(num_iter):
        x_history.append(x.copy())  # 将当前的x添加到历史列表中
        grad = numerical_gradient(f, x)  # 计算函数f在x处的梯度
        x -= lr * grad  # 沿着梯度的反方向更新x
    return x, np.array(x_history)


# 定义目标函数
def f(x):
    return x[0] ** 2 + x[1] ** 2  # 这是一个简单的二次函数，最小值在(0, 0)

if __name__ == '__main__':
    # 1. 定义初始值
    init_x = np.array([-3.0, 4.0])  # 从(-3, 4)开始进行梯度下降
    # 2. 定义超参数
    lr = 0.1  # 学习率
    num_iter = 20  # 迭代次数
    # 3. 使用梯度下降法计算最小值点
    x, x_history = gradient_desent(f, init_x, lr, num_iter)
    print("最小值点：", x)

    # 绘图
    plt.plot([-5, 5], [0, 0], '--b')  # 绘制对角线
    plt.plot([0, 0], [-5, 5], '--b')  # 绘制对角线
    plt.scatter(x_history[:, 0], x_history[:, 1], color='red')  # 绘制x历史点
    plt.xlim(-3.5, 3.5)  # 设置x轴范围
    plt.ylim(-3.5, 3.5)  # 设置y轴范围
    plt.xlabel("x[0]")  # x轴标签
    plt.ylabel("x[1]")  # y轴标签
    plt.show()