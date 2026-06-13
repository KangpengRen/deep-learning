import numpy as np

def step_function(x):
    """
    阶跃函数，张量操作
    """
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    """
    Sigmoid函数
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Tash函数
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    """
    ReLU函数
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax函数

    Args:
        x (_type_): 二维矩阵
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).T
    # 溢出处理策略
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def identify(x):
    """
    恒等函数
    """
    return x


def mean_squared_error(y, t):
    """
    均方误差
    """
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    """
    交叉熵误差
    """
    # 如果y是1维的，转化为2维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据是one-hot编码的情况下，转换为正确标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-10)) / batch_size

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print("x = ", x)
    print("step_function(x) = ", step_function(x))
    print("sigmoid(x) = ", sigmoid(x))
    # print("tanh(x) = ", tanh(x))
    print("tanh(x) = ", np.tanh(x))
    print("relu(x) = ", relu(x))
    print("softmax(x) = ", softmax(x))

    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [-1, -2, -3]])
    print("softmax(X) = ", softmax(X))