import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax
from config import DATA_DIR


# 读取数据
def get_data():
    # 1. 从文件加载数据集
    data = pd.read_csv(DATA_DIR / "train.csv")
    # 2. 划分数据集
    X = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # 3. 特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


# 初始化神经网络
def init_network():
    # 直接从文件中加载字典对象(网络结构、权重)
    network = joblib.load(DATA_DIR / "nn_sample")
    return network


# 前向传播
def froward(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y


# 主流程
if __name__ == "__main__":
    # 1. 获取数据
    _, x, _, y = get_data()
    print(x.shape)
    print(y.shape)

    # 2. 创建模型（加载参数）
    network = init_network()

    # # 3. 前向传播（测试）
    # y_prob = froward(network, x)
    # print(y_prob.shape)

    # # 4. 将分类概率转化为分类标签
    # y_pred = np.argmax(y_prob, axis=1)

    # 定义变量
    batch_size = 64
    accuary_cnt = 0
    n = x.shape[0]
    # 3. 循环迭代，分批次前向传播，累计预测正确的数量
    for i in range(0, n, batch_size):
        # 取出当前批次的数据
        x_batch = x[i : i + batch_size]
        # 前向传播
        y_batch_prob = froward(network, x_batch)
        # 将分类概率转化为分类标签
        y_batch_pred = np.argmax(y_batch_prob, axis=1)
        # 累计预测正确的数量
        accuary_cnt += np.sum(y_batch_pred == y[i : i + batch_size])

    # 5. 计算准确率
    accuracy = accuary_cnt / n
    print(f"Accuracy: {accuracy:.4f}")

    # 查看神经网络形状
    print("W1 shape:", network["W1"].shape)
    print("b1 shape:", network["b1"].shape)
    print("W2 shape:", network["W2"].shape)
    print("b2 shape:", network["b2"].shape)
    print("W3 shape:", network["W3"].shape)
    print("b3 shape:", network["b3"].shape)
