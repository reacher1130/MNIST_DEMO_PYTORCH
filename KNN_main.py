from torch import nn
from dataloader import *
from KNN_utils import *

if __name__ == '__main__':
    batch_size = 64
    train_data, test_data = load_data_mnist(batch_size)

    x_train = train_data.dataset.data.numpy()
    # 归一化处理
    mean_image = getXmean(x_train)
    x_train = centralized(x_train, mean_image)
    y_train = train_data.dataset.targets.numpy()
    # 对测试数据处理，取前num_test个测试数据
    num_test = 1000
    x_test = train_data.dataset.data[:num_test].numpy()
    mean_image = getXmean(x_test)
    x_test = centralized(x_test, mean_image)
    y_test = train_data.dataset.targets[:num_test].numpy()

    print("train_data:", x_train.shape)
    print("train_label:", len(y_train))
    print("test_data:", x_test.shape)
    print("test_labels:", len(y_test))

    # 利用KNN计算识别率
    for k in range(1, 6, 2):  # 不同K值计算识别率
        classifier = Knn()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(k, 'M', x_test)
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / num_test
        print('Got %d / %d correct when k= %d => accuracy: %f' % (num_correct, num_test, k, accuracy))
