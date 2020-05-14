import os

import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from sklearn.utils import shuffle

from km.graph import Graph
from km.node import Constant, CrossEntropyWithSoftMax, Linear, PlaceHolder, Sigmoid, Softmax, Variable
from km.optimizer import GradientDescent
from km.nn import normal
from km.session import Session

import numpy as np

data_folder = r'/Users/noone/development/python/data/notMNIST'
pickle_file = os.path.join(data_folder, 'notMNIST.pickle')
large_folder = os.path.join(data_folder, 'notMNIST_large')
small_folder = os.path.join(data_folder, 'notMNIST_small')

train_folders = [os.path.join(large_folder, d) for d in os.listdir(large_folder)]
test_folders = [os.path.join(small_folder, d) for d in os.listdir(small_folder)]

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_labels = 10


def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        return (train_dataset, train_labels), (test_dataset, test_labels), (valid_dataset, valid_labels)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0]


def nomnist():
    # 加载数据集文件，数据集文件处理详见notmnist_dataset.py
    train_data, test_data, valid_data = load_data(pickle_file)
    train_dataset = train_data[0]
    train_labels = train_data[1]
    test_dataset = test_data[0]
    test_labels = test_data[1]
    valid_dataset = valid_data[0]
    valid_labels = valid_data[1]
    # 将图片转换为一维数组
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)

    batch_size = 128
    graph = Graph()
    graph.as_default()
    # 初始化参数
    n_features = image_size * image_size  # X特征数
    n_hidden = 1024  # 隐藏层个数
    X = PlaceHolder(name='X', shape=(batch_size, n_features))
    y = PlaceHolder(name='y', shape=(batch_size, num_labels))
    W1 = Variable(normal((n_features, n_hidden), scale=1), name='W1')
    b1 = Variable(np.zeros(n_hidden), name='b1')
    W2 = Variable(normal((n_hidden, num_labels), scale=1), name='W2')
    b2 = Variable(np.zeros(num_labels), name='b2')

    # 定义训练模型
    l1 = Linear(X, W1, b1, name='l1')
    h1 = Sigmoid(l1, name='h1')
    l2 = Linear(h1, W2, b2, name='l2')
    loss = CrossEntropyWithSoftMax(l2, y, name='loss')
    # 定义优化器
    optimizer = GradientDescent(loss, learning_rate=0.001, name='sgd')
    # 输出训练结果
    train_prediction = Softmax(l2, name='output')
    # 定义测试模型
    X_test = Constant(test_dataset, name='X_test')
    l1_test = Linear(X_test, W1, b1, name='l1_test')
    h1_test = Sigmoid(l1_test, name='h1_test')
    l2_test = Linear(h1_test, W2, b2, name='l2_test')
    test_prediction = Softmax(l2_test, name='output_test')
    # 训练
    epoch = 1
    losses = []
    with Session.session(graph) as sess:
        for n in range(epoch):
            data, label = shuffle(train_dataset, train_labels)
            n_step = len(data) // batch_size + 1
            for i in range(n_step):
                b = i * batch_size
                e = b + batch_size
                if e > len(data):
                    b = -batch_size
                    e = len(data)
                batch_dataset = data[b:e]
                batch_labels = label[b:e]
                _, los, predictions = sess.run([optimizer, loss, train_prediction],
                                               feed_dict={X: batch_dataset, y: batch_labels})
                print('Epoch: {}, step: {}, loss={:.3f}, accuracy={:.2f}%'
                      .format(n + 1, i + 1, los, accuracy(predictions, batch_labels)))
                losses.append(los)
        print("Test accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), test_labels)))
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    nomnist()
