import time

from km.graph import Graph
from km.node import *
from km.nn import *
from km.optimizer import *
from km.session import Session
from examples.jay_lyrics_dataset import *
import numpy as np


def init_rnn_state(batch_size, num_hiddens):
    """ 获取初始状态 """
    return np.zeros((batch_size, num_hiddens))


def rnn_scratch():
    # 加载数据
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    # 初始化参数
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2

    # 创建计算图
    graph = Graph()
    graph.as_default()
    # 输入
    X = PlaceHolder(name='X', shape=(None, num_inputs))
    Y = PlaceHolder(name='Y', shape=(None, num_outputs))
    state = PlaceHolder(name='state', shape=(None, num_hiddens))  # 初始状态
    # 隐藏层参数
    W1 = Variable(normal((num_inputs, num_hiddens), scale=0.01), name='W1')  # 输入层参数
    Wt = Variable(normal((num_hiddens, num_hiddens), scale=0.01), name='Wt')  # 上一时间步参数
    b1 = Variable(np.zeros(num_hiddens), name='b1')
    # 输出层参数
    W2 = Variable(normal((num_hiddens, num_outputs), scale=0.01), name='W2')
    b2 = Variable(np.zeros(num_outputs), name='b2')
    # 定义模型
    dot_X_W1 = Dot(X, W1, name='dot_X_W1')
    dot_H_Wt = Dot(state, Wt, name='dot_H_Wt')
    add_XW_HW = Add(dot_X_W1, dot_H_Wt, name='add_XW_HW')
    L1 = Add(add_XW_HW, b1, name='L1')
    H = Tanh(L1, name='H')
    Y_hat = Linear(H, W2, b2, name='Y_hat')
    loss = CrossEntropyWithSoftMax(Y_hat, Y, name='loss')
    # 定义优化器
    optimizer = GradientDescent(loss, learning_rate=lr, grad_clipping=False, theta=clipping_theta, name='sgd')

    with Session.session(graph) as sess:
        for epoch in range(num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            # 使用相邻采样，在epoch开始时初始化隐藏状态
            H_t = init_rnn_state(batch_size, num_hiddens)
            data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
            for batch_data, batch_labels in data_iter:
                inputs = to_onehot(batch_data, vocab_size)
                for x, y in zip(inputs, batch_labels.T):
                    _, l, H_t = sess.run([optimizer, loss, H], feed_dict={X: x, Y: y.reshape(-1, 1), state: H_t})
                    l_sum += l
                n += batch_labels.size
            # 通常使用困惑度（perplexity）来评价语言模型的好坏。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，
            # 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
            # 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
            # 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
            # 任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小vocab_size。
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, l_sum / n, time.time() - start))


if __name__ == '__main__':
    rnn_scratch()
    from mxnet.gluon import loss as gloss

    gloss.SoftmaxCrossEntropyLoss()
