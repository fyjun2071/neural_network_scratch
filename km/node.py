import numpy as np

from . import graph
from .nn import softmax, sigmoid


class Node:
    """
    计算图节点类基类
    """

    def __init__(self, parents, name='Node'):
        self.name = name
        self.value = None   # 节点的值，np.matrix类型
        self.parents = parents
        self.children = []
        self.gradient = None  # 梯度
        for node in self.parents:
            node.children.append(self)  # 建立节点之间的连接

        # 计算图对象，默认为全局对象default_graph
        self.graph = graph.default_graph
        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def forward(self):
        """
        前向传播计算
        """
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute_value()
        return self.value

    def backward(self, result):
        """
        反向传播，链式法则
        """
        if self.gradient is None:
            if self is result:
                # 最后的损失函数节点
                self.gradient = np.mat(np.eye(self.dimension()))
            else:
                self.gradient = np.mat(np.zeros((result.dimension(), self.dimension())))
                for child in self.get_children():
                    if child.value is not None:
                        self.gradient += child.backward(result) * child.compute_grad(self)
        return self.gradient

    def compute_value(self):
        """
        前向传播计算本节点的值
        """
        raise NotImplementedError

    def compute_grad(self, parent):
        """
        反向传播，计算父节点对本节点的梯度
        """
        raise NotImplementedError

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def dimension(self):
        """
        返回本节点的值的向量维数
        """
        return self.value.shape[0] if self.value is not None else None

    def clear_gradient(self):
        self.gradient = None

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """
        self.value = None
        if recursive:
            for node in self.children:
                node.reset_value()

    def __repr__(self):
        return '{}:{}'.format(self.__class__.__name__, self.name)


class Variable(Node):
    """
    变量节点，np.matrix类型，列向量
    """

    def __init__(self, dim, init=False, trainable=True, name='Variable'):
        super().__init__([], name)
        self.dim = dim
        self.trainable = trainable

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, (self.dim, 1)))

    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == (self.dim, 1)
        # 本节点的值被改变，重置所有下游节点的值
        self.reset_value()
        self.value = value

    def compute_value(self):
        pass

    def compute_grad(self, parent):
        pass


class Add(Node):
    """
    向量加法
    """

    def __init__(self, input1, input2, name='Add'):
        super().__init__(parents=[input1, input2], name=name)
        self.input1 = input1
        self.input2 = input2

    def compute_value(self):
        assert self.input1.dimension() == self.input2.dimension()
        self.value = self.input1.value + self.input2.value

    def compute_grad(self, parent):
        return np.mat(np.eye(self.dimension()))  # 向量之和对其中任一个向量的雅可比矩阵是单位矩阵


class Dot(Node):
    """
    向量内积
    """

    def __init__(self, input1, input2, name='Dot'):
        super().__init__(parents=[input1, input2], name=name)
        self.input1 = input1
        self.input2 = input2

    def compute_value(self):
        assert self.input1.dimension() == self.input2.dimension()
        self.value = self.input1.value.T * self.input2.value

    def compute_grad(self, parent):
        if parent is self.input1:
            return self.input2.value.T
        else:
            return self.input1.value.T


class Vectorize(Node):
    """
    将多个父节点组装成一个向量
    """

    def __init__(self, parents, name='Vectorize'):
        super().__init__(parents=parents, name=name)
        self.parents = parents

    def compute_value(self):
        assert len(self.parents) > 0
        self.value = np.mat(np.array([node.value for node in self.parents])).T  # 将本节点的父节点的值列成向量

    def compute_grad(self, parent):
        return np.mat([node is parent for node in self.parents]).astype(np.float).T


class Sigmoid(Node):
    def __init__(self, node, name='Sigmoid'):
        super().__init__(parents=[node], name=name)
        self.x = node

    def compute_value(self):
        self.value = np.mat(sigmoid(self.x.value))

    def compute_grad(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class ReLU(Node):
    def __init__(self, node, name='ReLU'):
        super().__init__(parents=[node], name=name)
        self.x = node

    def compute_value(self):
        v = self.x.value
        self.value = np.mat(np.where(v > 0, v, 0.1 * v))

    def compute_grad(self, parent):
        return np.diag(np.where(self.x.value.A1 > 0, 1, 0.1))


class MSE(Node):
    def __init__(self, y_true, y_hat, name='MSE'):
        super().__init__(parents=[y_true, y_hat], name=name)
        self.y_true = y_true
        self.y_hat = y_hat
        self.diff = None

    def compute_value(self):
        self.diff = self.y_true.value - self.y_hat.value
        self.value = np.mat(0.5 * np.mean(self.diff ** 2))

    def compute_grad(self, parent):
        if parent is self.y_true:
            return (self.diff / self.dimension()).T
        else:
            return -(self.diff / self.dimension()).T


class Softmax(Node):
    def __init__(self, x, name='Softmax'):
        super().__init__(parents=[x], name=name)
        self.x = x

    def compute_value(self):
        self.value = softmax(self.x.value)

    def compute_grad(self, parent):
        """
        我们不实现SoftMax节点的导数，训练时使用CrossEntropyWithSoftMax节点
        """
        return np.mat(np.eye(self.dimension()))  # 无用


class CrossEntropyWithSoftMax(Node):
    """
    交叉熵损失函数
    """

    def __init__(self, x, labels, name='CrossEntropyWithSoftMax'):
        super().__init__(parents=[x, labels], name=name)
        self.x = x
        self.labels = labels
        self.y = None

    def compute_value(self):
        self.y = softmax(self.x.value)
        self.value = np.mat(-np.sum(np.multiply(self.labels.value, np.log(self.y + 1e-10))))

    def compute_grad(self, parent):
        if parent is self.x:
            return (self.y.value - self.labels.value).T
        else:
            return (-np.log(self.y.value)).T
