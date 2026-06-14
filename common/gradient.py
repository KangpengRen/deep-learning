import numpy as np

# 数值微分
def numerical_diff(f, x):
    # x为标量
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)  # 中心差分

# 使用数值微分求梯度
def _numerical_gradient(f, x):
    # x为向量
    h = 1e-4
    grad = np.zeros_like(x)
    # 遍历x中的特征xi
    for i in range(x.size):
        tmp = x[i]  # 备份原值
        x[i] = tmp + h
        fxh1 = f(x)  # f(x+h)
        x[i] = tmp - h
        fxh2 = f(x)  # f(x-h)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp  # 恢复原值
    return grad

def numerical_gradient(f, X):
    # X为矩阵
    if X.ndim == 1:  # 如果X是一维向量
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient(f, x)
        return grad