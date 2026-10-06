import data
import train
import plot_img
import numpy as np
from typing import List, Tuple


def run_sepal() -> None:
    """原实验：只使用萼片长度、宽度。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_data()

    clf = train.train_and_evaluate(x_train, y_train, x_test, y_test, 'Sepal')

    # 绘图
    plot_img.draw(clf, x, y, x_test)


def run_petal() -> None:
    """新增实验：只使用花瓣长度、宽度。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_petal_data()
    clf = train.train_and_evaluate(x_train, y_train, x_test, y_test, 'Petal')
    plot_img.draw_petal(clf, x, y, x_test)


def run_all_features() -> None:
    """新增实验：使用全部四个特征，展示测试集预测结果。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_all_data()
    clf = train.train_and_evaluate(x_train, y_train, x_test, y_test, 'All features')
    plot_img.draw_all_features(clf, x_test, y_test)


def _find_best_c(
        x_train, y_train, x_test, y_test, feature_name: str, c_list: List[float]
        ) -> Tuple[float, float]:
    """遍历候选 C，返回测试集准确率最高的 C 和对应准确率。"""
    if not c_list:
        raise ValueError('c_list 不能为空。')

    best_c = c_list[0]
    best_accuracy = -1.0
    for c in c_list:
        _, test_accuracy = train.train_and_evaluate_change_c(
            x_train, y_train, x_test, y_test, feature_name, c)
        # 使用 > 而非 >=：准确率相同时，保留 c_list 中较早的较小 C。
        if test_accuracy > best_accuracy:
            best_c = c
            best_accuracy = test_accuracy

    print('\n%s best C: %g, test accuracy: %.3f'
          % (feature_name, best_c, best_accuracy))
    return best_c, best_accuracy


def run_sepal_change_c(c_list: List[float]) -> Tuple[float, float]:
    """原实验：只使用萼片长度、宽度。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_data()

    return _find_best_c(x_train, y_train, x_test, y_test, 'Sepal', c_list)


def run_petal_change_c(c_list: List[float]) -> Tuple[float, float]:
    """新增实验：只使用花瓣长度、宽度。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_petal_data()

    return _find_best_c(x_train, y_train, x_test, y_test, 'Petal', c_list)


def run_all_features_change_c(c_list: List[float]) -> Tuple[float, float]:
    """新增实验：使用全部四个特征，展示测试集预测结果。"""
    x, y, x_train, x_test, y_train, y_test = data.iris_load_all_data()

    return _find_best_c(x_train, y_train, x_test, y_test, 'All features', c_list)


if __name__ == "__main__":
    # 保持模型参数和划分方式不变，依次比较三组特征。
    # 使用交互式绘图后端时，关闭当前图窗后继续下一组实验。
    """run_sepal()
    run_petal()
    run_all_features()
    """
    c_list = np.linspace(0.001, 30, 300).tolist()
    best_c_sepal, best_accuracy_sepal = run_sepal_change_c(c_list)
    best_c_petal, best_accuracy_petal = run_petal_change_c(c_list)
    best_c_all, best_accuracy_all = run_all_features_change_c(c_list)

    print(
        f"\nbest c in sepal = {best_c_sepal}, accuracy = {best_accuracy_sepal}\n",
        f"best c in petal = {best_c_petal}, accuracy = {best_accuracy_petal}\n",
        f"best c in all feature = {best_c_all}, accuracy = {best_accuracy_all}\n"
    )
