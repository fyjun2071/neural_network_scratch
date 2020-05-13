import time

from km.graph import Graph
from km.node import *
from km.nn import *
from km.optimizer import *
from km.session import Session
from examples.jay_lyrics_dataset import *
import numpy as np


def init_gru_state(batch_size, num_hiddens):
    """ 初始状态 """
    return np.zeros((batch_size, num_hiddens))


class HCandidate(Node):
    """ 候选隐藏状态的线性输出 """

    def __init__(self, X, W_xh, R, H, W_hh, b_h, name='HCandidate'):
        super().__init__(inputs=[X, W_xh, R, H, W_hh, b_h], name=name)
        self.X = X
        self.W_xh = W_xh
        self.R = R
        self.H = H
        self.W_hh = W_hh
        self.b_h = b_h
        self.RH = None

    def forward(self):
        X = self.X.value
        W_xh = self.W_xh.value
        R = self.R.value
        H = self.H.value
        W_hh = self.W_hh.value
        b_h = self.b_h.value
        RH = R * H
        self.RH = RH
        self.value = np.dot(X, W_xh) + np.dot(RH, W_hh) + b_h

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.W_xh] += np.dot(self.X.value.T, grad_cost)
            self.gradients[self.R] += np.dot(grad_cost, self.W_hh.value.T) * self.H.value
            self.gradients[self.W_hh] += np.dot(self.RH.T, grad_cost)
            self.gradients[self.b_h] += np.sum(grad_cost * 1, axis=0, keepdims=False)


class HState(Node):
    """ 隐藏状态输出 """

    def __init__(self, Z, H, H_tilda, name='HState'):
        super().__init__(inputs=[Z, H, H_tilda], name=name)
        self.Z = Z
        self.H = H
        self.H_tilda = H_tilda
        self.Z_1 = None

    def forward(self):
        Z = self.Z.value
        H = self.H.value
        H_tilda = self.H_tilda.value
        Z_1 = 1 - Z
        self.Z_1 = Z_1
        self.value = Z * H + Z_1 * H_tilda

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.Z] += grad_cost * (self.H.value - self.H_tilda.value)
            self.gradients[self.H_tilda] += grad_cost * self.Z_1


def gru_scratch():
    # 加载数据
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2

    # 创建计算图
    graph = Graph()
    graph.as_default()
    # 输入
    X = PlaceHolder(name='X', shape=(None, num_inputs))
    Y = PlaceHolder(name='Y', shape=(None, num_outputs))
    state = PlaceHolder(name='state', shape=(None, num_hiddens))  # 初始状态
    H = state  # 上一时间步隐藏状态
    # 更新门参数
    W_xz = Variable(normal((num_inputs, num_hiddens), scale=0.01), name='W_xz')
    W_hz = Variable(normal((num_hiddens, num_hiddens), scale=0.01), name='W_hz')
    b_z = Variable(np.zeros(num_hiddens), name='b_z')
    # 充值门参数
    W_xr = Variable(normal((num_inputs, num_hiddens), scale=0.01), name='W_xr')
    W_hr = Variable(normal((num_hiddens, num_hiddens), scale=0.01), name='W_hr')
    b_r = Variable(np.zeros(num_hiddens), name='b_r')
    # 候选隐藏状态参数
    W_xh = Variable(normal((num_inputs, num_hiddens), scale=0.01), name='W_xh')
    W_hh = Variable(normal((num_hiddens, num_hiddens), scale=0.01), name='W_hh')
    b_h = Variable(np.zeros(num_hiddens), name='b_h')
    # 输出层参数
    W_hq = Variable(normal((num_hiddens, num_outputs), scale=0.01), name='W_hz')
    b_q = Variable(np.zeros(num_outputs), name='b_z')
    # 定义模型
    # 更新门
    L_Z = PolyLinear([X, H], [W_xz, W_hz], b_z, name='L_Z')
    Z = Sigmoid(L_Z, name='Z')
    # 重置门
    L_R = PolyLinear([X, H], [W_xr, W_hr], b_r, name='L_R')
    R = Sigmoid(L_R, name='R')
    # 候选隐藏状态
    L_H_tilda = HCandidate(X, W_xh, R, H, W_hh, b_h, name='L_H_tilda')
    H_tilda = Tanh(L_H_tilda, name='H_tilda')
    # 隐藏状态
    H_t = HState(Z, H, H_tilda, name='H_t')
    # 输出
    Y_hat = Linear(H_t, W_hq, b_q, name='Y_hat')
    # 定义loss
    loss = CrossEntropyWithSoftMax(Y_hat, Y, name='loss')
    # 定义优化器
    optimizer = GradientDescent(loss, learning_rate=lr, name='sgd')

    with Session.session(graph) as sess:
        for epoch in range(num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            # 使用相邻采样，在epoch开始时初始化隐藏状态
            H_value = init_gru_state(batch_size, num_hiddens)
            data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
            for batch_data, batch_labels in data_iter:
                inputs = to_onehot(batch_data, vocab_size)
                for x, y in zip(inputs, batch_labels.T):
                    _, l, H_value = sess.run([optimizer, loss, H],
                                             feed_dict={X: x, Y: y.reshape(-1, 1), state: H_value})
                    l_sum += l
                n += batch_labels.size
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, l_sum / n, time.time() - start))


if __name__ == '__main__':
    gru_scratch()
