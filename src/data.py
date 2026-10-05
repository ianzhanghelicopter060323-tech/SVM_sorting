import numpy as np
from sklearn import svm 
from sklearn import model_selection


def iris_type(s) :
    """ 标签的 str -> int 类型转换
        神人编程语言， iris_type()明确数据诶性converters参数报错 
    """
    it = {b'Iris-setosa':0, b'Iris-versicolor':1,b'Iris-virginica':2} 
    return it[s]


def iris_load_data() -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]" :
    """ 数据集与测试集的准备 """

    # 加载数据集
    data = np.loadtxt(
        "/home/ianzhang/ai_foundation_prog/SVM_sorting/dataset/iris.data", # 数据集路径
        dtype=float, # 数据类型
        delimiter=",", # 分割符
        converters={4: iris_type} # 取 iris.data的第五列（标签string）, 调用 iris_type函数将其转换为 int
    )

    # 数据分割
    # 前四列是x， 后一列是y，分割
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :2] # 取前两列

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, 
        random_state=1, # 固定随机种子
        test_size=0.2   # 20%样本作为测试集， 80%样本作为训练集
        )
    return x, y, x_train, x_test, y_train, y_test

