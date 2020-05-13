from .node import Variable


class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表
        self.params = []  # 变量节点
        self.session = None

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)
        if isinstance(node, Variable):
            self.params.append(node)

    def clear_gradients(self):
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_gradients()

    def reset_value(self):
        """
        重置图中全部节点的值
        """
        for node in self.nodes:
            node.reset_value(False)  # 每个节点不递归清除自己的子节点的值

    def as_default(self):
        global default_graph
        default_graph = self

    def set_session(self, session):
        self.session = session


# 全局默认计算图
default_graph = Graph()
