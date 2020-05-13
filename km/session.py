from contextlib import contextmanager

from .graph import default_graph
from .node import PlaceHolder


class Session:

    def __init__(self, graph=default_graph):
        self.graph = graph

    def eval(self, node):
        """ 回溯到起点开始前向传播 """
        for n in node.inputs:
            self.eval(n)
        node.forward()
        return node.value

    def run(self, fetches, feed_dict=None):
        # 设置数据
        if feed_dict:
            for k, v in feed_dict.items():
                if isinstance(k, PlaceHolder):
                    k.value = v
        # 对每个节点执行前向计算
        fetches_result = []
        for n in fetches:
            fetches_result.append(self.eval(n))
        return tuple(fetches_result)

    @staticmethod
    @contextmanager
    def session(graph):
        sess = Session(graph)
        graph.set_session(sess)
        yield sess
        # 清理计算图中数据
        graph.reset_value()
        graph.clear_gradients()
