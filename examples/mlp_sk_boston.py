from sklearn.datasets import load_boston
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from km.graph import Graph
from km.node import Linear, MSE, PlaceHolder, Sigmoid, Variable
from km.optimizer import GradientDescent
from km.nn import normal, standardization
from km.session import Session

import numpy as np


def mlp():
    # 载入数据
    data = load_boston()
    dataset = data['data']
    labels = data['target']
    graph = Graph()
    graph.as_default()
    # 初始化参数
    n_features = dataset.shape[1]  # X特征数
    n_hidden = 100  # 隐藏层个数
    X = PlaceHolder(name='X')
    y = PlaceHolder(name='y')
    W1 = Variable(normal((n_features, n_hidden), scale=1), name='W1')
    b1 = Variable(np.zeros(n_hidden), name='b1')
    W2 = Variable(normal((n_hidden, 1), scale=1), name='W2')
    b2 = Variable(np.zeros(1), name='b2')

    # 定义模型
    l1 = Linear(X, W1, b1, name='l1')
    h1 = Sigmoid(l1, name='h1')
    yhat = Linear(h1, W2, b2, name='yhat')
    loss = MSE(y, yhat, name='loss')

    epoch = 5001
    batch_size = 256
    optimizer = GradientDescent(loss, learning_rate=0.001, name='sgd')
    losses = []
    with Session.session(graph) as sess:
        for n in range(epoch):
            data, label = shuffle(dataset, labels)
            loss_sum = 0
            n_step = len(dataset) // batch_size + 1
            for i in range(n_step):
                b = i * batch_size
                e = b + batch_size
                if e > len(dataset):
                    b = -batch_size
                    e = len(dataset)
                batch_dataset = standardization(data[b:e])
                batch_labels = label[b:e]
                _, los = sess.run([optimizer, loss], feed_dict={X: batch_dataset, y: batch_labels})
                # print('step: {}, loss = {:.3f}'.format(i+1, los))
                loss_sum += los
            if n % 100 == 0:
                print('Epoch: {}, loss = {:.3f}'.format(n + 1, loss_sum / n_step))
                losses.append(los)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    mlp()
