import data
import train
import plot_img

if __name__ == "__main__" :
    x, y, x_train, x_test, y_train, y_test = data.iris_load_data()

    # 训练
    clf = train.classifier()
    train.train(clf, x_train=x_train, y_train=y_train)
    print("=========================== training finished ===========================")

    # 输出准确率
    train.print_accuracy(clf, x_train, y_train, x_test, y_test)

    # 绘图
    plot_img.draw(clf, x, y, x_test)