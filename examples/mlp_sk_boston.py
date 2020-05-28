from sklearn.datasets import load_boston
from torch.utils.data import DataLoader, Dataset, random_split
from km.node import Add, Dot, MSE, ReLU, Variable, Vectorize
from km.optimizer import GradientDescent

# 载入数据
boston = load_boston()
data = boston['data']
labels = boston['target']


class BostonDataset(Dataset):
    """
    合并数据与标签
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)


# 划分数据集
batch_size = 32
ratio = 0.7
train_size = int(data.shape[0] * ratio)
test_size = data.shape[0] - train_size
train, test = random_split(BostonDataset(data, labels), [train_size, test_size])
train_data_iter = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

# 模型
x_dim = data.shape[1]
hidden_size = 100
x = Variable(x_dim, trainable=False)
# 单个神经元参数
w1 = Variable(x_dim, trainable=True)
b1 = Variable(1, trainable=True)
# 将多个神经元组装成隐藏
hidden = []
for i in range(hidden_size):
    hidden.append(Add(Dot(w1, x), b1))
h = ReLU(Vectorize(hidden))   # 将单个神经元组成的隐藏层输出组装成向量
# 输出层
w2 = Variable(hidden_size, trainable=True)
b2 = Variable(1, trainable=True)
y_hat = Add(Dot(w2, h), b2)

# 损失函数
y = Variable(1, trainable=False)
loss = MSE(y, y_hat)

# 优化器
optimizer = GradientDescent(loss, batch_size=batch_size)
