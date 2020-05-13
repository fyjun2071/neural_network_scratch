import numpy as np


def normal(shape, loc=0.0, scale=1.0):
    return np.random.normal(loc=loc, scale=scale, size=shape)


def standardization(data, axis=0):
    mu = np.mean(data, axis=axis)
    sigma = np.std(data, axis=axis, ddof=1)
    return (data - mu) / sigma


def softmax(X):
    """y_i = (x_i - max(x)) / sum(x - max(x))"""
    _X = X - np.max(X, axis=1).reshape(-1, 1)
    ep = np.exp(_X)
    return ep / np.sum(ep, axis=1).reshape(-1, 1)


def sigmoid(x):
    z = np.where(-x > 1e2, 1e2, -x)
    return 1.0 / (1 + np.exp(z))


def dropout(X, drop_prob):
    """
    丢弃法
    :param X:
    :param drop_prob: 丢弃概率
    :return:
    """
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return X.zeros_like()
    mask = np.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob


def grad_clipping(params, theta):
    """
    梯度裁剪，处理梯度爆炸
    :param params: Variable类型的参数节点
    :param theta:
    :return:
    """
    norm = np.array([0])
    for param in params:
        norm += (param.gradients[param] ** 2).sum()
    norm = np.sqrt(norm)
    if norm > theta:
        for param in params:
            param.gradients[param][:] *= theta / norm


def to_onehot(X, size):
    """ onehot编码 """
    return [np.eye(size)[x] for x in X.T]
