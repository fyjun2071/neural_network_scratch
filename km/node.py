import numpy as np

from . import graph
from .nn import softmax, sigmoid


class Node:
    """
    计算图节点类基类
    """

    def __init__(self, inputs=None, name=''):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.name = name
        self.value = None
        self.outputs = []
        self.gradients = {}  # 梯度
        self.graph = graph.default_graph  # 计算图对象，默认为全局对象default_graph
        self.params = []

        for node in self.inputs:
            node.outputs.append(self)  # 建立节点之间的连接

        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def forward(self):
        """
        前向传播计算本节点的值
        """
        raise NotImplemented

    def backward(self):
        """
        反向传播，计算结果节点对本节点的梯度
        """
        raise NotImplemented

    def eval(self):
        return self.graph.session.eval(self)

    def clear_gradients(self):
        self.gradients.clear()

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """
        self.value = None
        if recursive:
            for o in self.outputs:
                o.reset_value()

    def __repr__(self):
        return '{}:{}'.format(self.__class__.__name__, self.name)


class PlaceHolder(Node):
    def __init__(self, shape=None, name='PlaceHolder'):
        super().__init__([], name)
        self.shape = shape

    def forward(self):
        pass

    def backward(self):
        pass


class Constant(Node):
    def __init__(self, value, name='Constant'):
        super().__init__([], name)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        pass


class Variable(Node):
    def __init__(self, value=None, name='Variable'):
        super().__init__([], name)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self] = grad_cost


class Add(Node):
    def __init__(self, input_1, input_2, name='Dot'):
        super().__init__(inputs=[input_1, input_2], name=name)
        self.input1_node = input_1
        self.input2_node = input_2

    def forward(self):
        self.value = self.input1_node.value + self.input2_node.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.input1_node] += np.sum(grad_cost * 1, axis=0, keepdims=False)
            self.gradients[self.input2_node] += np.sum(grad_cost * 1, axis=0, keepdims=False)


class Dot(Node):
    """ 点乘 """

    def __init__(self, input_1, input_2, name='Dot'):
        super().__init__(inputs=[input_1, input_2], name=name)
        self.input1_node = input_1
        self.input2_node = input_2

    def forward(self):
        self.value = np.dot(self.input1_node.value, self.input2_node.value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.input1_node] += np.dot(grad_cost, self.input2_node.value.T)
            self.gradients[self.input2_node] += np.dot(self.input1_node.value.T, grad_cost)


class Multi(Node):
    """ 数乘 """

    def __init__(self, input_1, input_2, name='Multi'):
        super().__init__(inputs=[input_1, input_2], name=name)
        self.input1_node = input_1
        self.input2_node = input_2

    def forward(self):
        self.value = self.input1_node.value * self.input2_node.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.input1_node] += grad_cost * self.input2_node.value
            self.gradients[self.input2_node] += grad_cost * self.input1_node.value


class Linear(Node):
    def __init__(self, X, W, b, name='Linear'):
        super().__init__(inputs=[X, W, b], name=name)
        self.x_node = X
        self.w_node = W
        self.b_node = b

    def forward(self):
        self.value = np.dot(self.x_node.value, self.w_node.value) + self.b_node.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            self.gradients[self.w_node] += np.dot(self.x_node.value.T, grad_cost)
            self.gradients[self.x_node] += np.dot(grad_cost, self.w_node.value.T)
            self.gradients[self.b_node] += np.sum(grad_cost * 1, axis=0, keepdims=False)


class PolyLinear(Node):
    """
    多元线性节点
    """

    def __init__(self, x_nodes, w_nodes, b_node, name='PolyLinear'):
        """
        :param x_nodes: 所有的X
        :param w_nodes: 所有的W
        :param b_node:
        :param name:
        """
        super().__init__(inputs=[*x_nodes, *w_nodes, b_node], name=name)
        self.x_nodes = x_nodes
        self.w_nodes = w_nodes
        self.b_node = b_node

    def forward(self):
        for x_node, w_node in zip(self.x_nodes, self.w_nodes):
            if self.value is None:
                self.value = np.dot(x_node.value, w_node.value)
            self.value += np.dot(x_node.value, w_node.value)
        self.value += self.b_node.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for node in self.outputs:
            if self not in node.gradients:
                continue
            grad_cost = node.gradients[self]
            for x_node, w_node in zip(self.x_nodes, self.w_nodes):
                self.gradients[w_node] += np.dot(x_node.value.T, grad_cost)
                self.gradients[x_node] += np.dot(grad_cost, w_node.value.T)
            self.gradients[self.b_node] += np.sum(grad_cost * 1, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node, name='Sigmoid'):
        super().__init__(inputs=[node], name=name)
        self.x_node = node
        self.partial = None

    def forward(self):
        self.value = sigmoid(self.x_node.value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        y = self.value
        self.partial = y * (1 - y)
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x_node] += self.partial * grad_cost


class ReLU(Node):
    def __init__(self, node, name='ReLU'):
        super().__init__(inputs=[node], name=name)
        self.x_node = node

    def forward(self):
        v = self.x_node.value
        self.value = np.where(v > 0, v, 0)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x_node] += np.where(self.value > 0, 1, 0) * grad_cost


class Tanh(Node):
    def __init__(self, node, name='Node'):
        super().__init__(inputs=[node], name=name)
        self.x_node = node

    def forward(self):
        self.value = 2 * sigmoid(2 * self.x_node.value) - 1

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        y = self.value
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.x_node] += (1 - y ** 2) * grad_cost


class MSE(Node):
    def __init__(self, y_true, y_hat, name='MSE'):
        super().__init__(inputs=[y_true, y_hat], name=name)
        self.y_true_node = y_true
        self.y_hat_node = y_hat
        self.diff = None

    def forward(self):
        y_true_flatten = self.y_true_node.value.reshape(-1, 1)
        y_hat_flatten = self.y_hat_node.value.reshape(-1, 1)
        self.diff = y_true_flatten - y_hat_flatten
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        n = self.y_hat_node.value.shape[0]
        self.gradients[self.y_true_node] = (2 / n) * self.diff
        self.gradients[self.y_hat_node] = (-2 / n) * self.diff


class Softmax(Node):
    def __init__(self, x, name='Softmax'):
        super().__init__(inputs=[x], name=name)
        self.x_node = x

    def forward(self):
        self.value = softmax(self.x_node.value)

    def backward(self):
        """
        我们不实现SoftMax节点的导数，训练时使用CrossEntropyWithSoftMax节点
        """
        self.gradients[self.x_node] = np.eye(self.x_node.value.shape)


class CrossEntropyWithSoftMax(Node):
    def __init__(self, x, labels, name='CrossEntropyWithSoftMax'):
        super().__init__(inputs=[x, labels], name=name)
        self.x_node = x
        self.labels = labels
        self.y = None

    def forward(self):
        self.y = softmax(self.x_node.value)
        self.value = np.mean(-np.sum(self.labels.value * np.log(self.y + 1e-10), axis=1))

    def backward(self):
        self.gradients[self.x_node] = self.y - self.labels.value
