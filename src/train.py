from typing import Tuple

import numpy as np
from sklearn import svm 


def classifier() -> svm.SVC :
    """ 构建分类器函数 """
    clf = svm.SVC(
        C=0.8, # 误分类惩罚系数，控制间隔大小与训练误差之间的权衡
        kernel='linear', # 可选内置核：linear、poly、rbf、sigmoid
        decision_function_shape='ovr' # 决策函数：计算分类分数；ovr本质上是ovo（两两比较）
    )
    return clf


def classifier_change_c(c_in: float) -> svm.SVC :
    """ 构建分类器函数， 传入C的值 """
    clf = svm.SVC(
        C=c_in, # 误分类惩罚系数，控制间隔大小与训练误差之间的权衡
        kernel='linear', # 可选内置核：linear、poly、rbf、sigmoid
        decision_function_shape='ovr' # 决策函数：计算分类分数；ovr本质上是ovo（两两比较）
    )
    return clf


def train(clf :svm.SVC, x_train: np.ndarray, y_train: np.ndarray) -> None :
    """ SVM训练函数 """
    clf.fit(x_train, y_train.ravel()) # ravel()将标签展开成一维数组


def train_and_evaluate(
        x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray, feature_name: str
        ) -> svm.SVC:
    """每组特征独立训练一个模型，使用相同参数并输出对应准确率。"""
    clf = classifier()
    train(clf, x_train, y_train)
    print('\n================ %s (%d features) ================'
          % (feature_name, x_train.shape[1]))
    print_accuracy(clf, x_train, y_train, x_test, y_test)
    return clf


def train_and_evaluate_change_c(
        x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray, feature_name: str,
        c_in: float
        ) -> Tuple[svm.SVC, float]:
    """每组特征独立训练一个模型, 使用相同参数并输出对应准确率。改变c实验"""

    clf = classifier_change_c(c_in)
    train(clf, x_train, y_train)
    print('\n================ %s (%d features, C=%g) ================'
          % (feature_name, x_train.shape[1], c_in))
    test_accuracy = print_accuracy(clf, x_train, y_train, x_test, y_test)
    return clf, test_accuracy


def show_accuracy(y_predict: np.ndarray, y_trian: np.ndarray, tip: str) -> float:
    """打印并返回预测结果相对于真实标签的准确率。"""
    acc = y_predict.ravel() == y_trian.ravel() # acc是一个bool型的数组
    accuracy = np.mean(acc)
    print('%s Accuracy:%.3f' %(tip, accuracy))
    return accuracy

    
def print_accuracy(
        clf: svm.SVC, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray
        ) -> float:
    """分别打印准确率，并返回测试集准确率。

        score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率 
    """

    print('training prediction:%.3f' %(clf.score(x_train, y_train.ravel()))) # 训练集准确率（隐式）
    print('test data prediction:%.3f' %(clf.score(x_test, y_test.ravel()))) # 测试集准确率（隐式）
    
    # 原始结果和预测结果进行对比 predict() 表示对x_train样本进行预测,返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'traing data') # 训练集准确率（显式）
    test_accuracy = show_accuracy(clf.predict(x_test), y_test, 'testing data')
    
    # ovr 返回各类别的决策分数，不是概率，也不是各分割面的几何距离
    # print('decision_function:\n', clf.decision_function(x_train))
    return test_accuracy
