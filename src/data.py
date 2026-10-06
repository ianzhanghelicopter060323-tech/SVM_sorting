from pathlib import Path

import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


def iris_type(s) :
    """将类别标签转换为整数，兼容 NumPy 传入的 bytes 和 str。"""
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    it = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    return it[s]


def iris_load_data() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]" :
    """准备萼片长度、宽度数据，保留原实验入口。"""
    return _load_features(slice(0, 2))


def iris_load_petal_data() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """准备花瓣长度、宽度数据。"""
    return _load_features(slice(2, 4))


def iris_load_all_data() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """准备全部四个特征的数据。"""
    return _load_features(slice(0, 4))


def iris_load_data_sigmoid() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """sigmoid 实验：准备标准化后的萼片长度、宽度。"""
    return _load_feature_sigmoid(slice(0, 2))


def iris_load_petal_data_sigmoid() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """sigmoid 实验：准备标准化后的花瓣长度、宽度。"""
    return _load_feature_sigmoid(slice(2, 4))


def iris_load_all_data_sigmoid() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """sigmoid 实验：准备标准化后的全部四个特征。"""
    return _load_feature_sigmoid(slice(0, 4))


def _load_features(feature_slice: slice) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """三组实验共用读取和划分逻辑，只改变输入特征。"""

    # 加载数据集
    data = np.loadtxt(
        Path(__file__).resolve().parent.parent / "dataset" / "iris.data",
        dtype=float, # 数据类型
        delimiter=",", # 分割符
        converters={4: iris_type} # 取 iris.data的第五列（标签string）, 调用 iris_type函数将其转换为 int
    )

    # 数据分割
    # 前四列是x， 后一列是y，分割
    x, y = np.split(data, (4,), axis=1)
    x = x[:, feature_slice]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, 
        random_state=1, # 相同行顺序和种子保证三组实验使用相同的训练/测试样本
        test_size=0.2   # 20%样本作为测试集， 80%样本作为训练集
        )
    return x, y, x_train, x_test, y_train, y_test

def _load_feature_sigmoid(feature_slice: slice) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """复用原数据划分，仅用训练集拟合标准化参数，返回顺序与原函数一致。"""
    x, y, x_train, x_test, y_train, y_test = _load_features(feature_slice)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # 测试集和全体绘图样本沿用训练集的均值、标准差，避免数据泄漏。
    x_test_scaled = scaler.transform(x_test)
    x_scaled = scaler.transform(x)
    # 三份特征处于同一标准化坐标系，标签和样本顺序保持不变。
    return x_scaled, y, x_train_scaled, x_test_scaled, y_train, y_test
