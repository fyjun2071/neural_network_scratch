

class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)

    def clear_gradient(self):
        """
        清除图中全部节点的梯度
        """
        for node in self.nodes:
            node.clear_gradient()

    def reset_value(self):
        """
        重置图中全部节点的值
        """
        for node in self.nodes:
            node.reset_value(False)  # 每个节点不递归清除自己的子节点的值

    def as_default(self):
        global default_graph
        default_graph = self


# 全局默认计算图
default_graph = Graph()
