import data
import train
import plot_img


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


if __name__ == "__main__":
    # 保持模型参数和划分方式不变，依次比较三组特征。
    # 使用交互式绘图后端时，关闭当前图窗后继续下一组实验。
    run_sepal()
    run_petal()
    run_all_features()
