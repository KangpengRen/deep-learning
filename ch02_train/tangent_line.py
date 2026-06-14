import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff

# 原函数 y = 0.01x^2 + 0.1x
def f(x):
    return x ** 2 * 0.01 + x * 0.1

# 切线方程函数，返回切线对应的函数
def tangent_line(f, x):
    y = f(x)    # 切点的函数值
    # 切点的切线斜率，利用数值微分求得
    k = numerical_diff(f, x)
    print("切点的切线斜率：", k)
    b = y - k * x  # 切线方程的截距
    return lambda t: k * t + b  # 切线方程

if __name__ == '__main__':
    # 定义画图的范围
    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)

    f_line = tangent_line(f, x=5)  # 求 x=5 处的切线函数
    y_line = f_line(x)

    plt.plot(x, y, label='f(x)')  # 原函数
    plt.plot(x, y_line, label='tangent line')  # 切线
    plt.show()