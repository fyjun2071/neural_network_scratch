from .node import Node
import numpy as np


class Optimizer(Node):
    """
    优化器基类
    """

    def __init__(self, target, name='Optimizer'):
        super().__init__(inputs=[target], name=name)
        self.target = target

    def forward(self):
        # for node in self.graph.nodes:
        #     node.forward()
        self.backward()

    def backward(self):
        inputs = list(self.inputs)
        while len(inputs) > 0:
            node = inputs.pop(0)
            node.backward()
            inputs += node.inputs
        self.update()

    def update(self):
        """
        抽象方法，利用梯度更新可训练变量
        """
        raise NotImplemented


class GradientDescent(Optimizer):
    """
    梯度下降优化器
    """

    def __init__(self, target, learning_rate=0.01, grad_clipping=False, theta=0, name='GradientDescent'):
        super().__init__(target, name=name)
        self.learning_rate = learning_rate
        self.grad_clipping = grad_clipping
        self.theta = theta

    def update(self):
        if self.grad_clipping:
            self._grad_clipping()
        for node in self.graph.params:
            gradient = node.gradients[node]
            node.value -= self.learning_rate * gradient

    def _grad_clipping(self):
        """ 梯度裁剪，处理梯度爆炸 """
        params = self.graph.params
        norm = 0
        for param in params:
            norm += (param.gradients[param] ** 2).sum()
        norm = np.sqrt(norm)
        if norm > self.theta:
            for param in params:
                param.gradients[param][:] *= self.theta / norm
