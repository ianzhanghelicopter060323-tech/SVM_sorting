import data
import train


if __name__ == "__main__" :
    x_train, x_test, y_train, y_test = data.iris_load_data()

    clf = train.classifier()
    train.train(clf, x_train=x_train, y_train=y_train)