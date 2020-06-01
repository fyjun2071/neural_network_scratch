from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from km.node import Add, Dot, MSE, ReLU, Variable, Vectorize
from km.optimizer import GradientDescent
import numpy as np

# 载入数据
boston = load_boston()
data = boston['data']
data = StandardScaler().fit_transform(data)
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
x = Variable(x_dim, trainable=False, name='x')
# 将多个神经元组装成隐藏
hidden = []
for i in range(hidden_size):
    hidden.append(Add(Dot(Variable(x_dim, init=True, name='w1'), x), Variable(1, init=True, name='b1')))
h = ReLU(Vectorize(hidden), name='h')   # 将单个神经元组成的隐藏层输出组装成向量
# 输出层
w2 = Variable(hidden_size, init=True, name='w2')
b2 = Variable(1, init=True, name='b2')
y_hat = Add(Dot(w2, h), b2, name='y_hat')

# 损失函数
y = Variable(1, trainable=False, name='y')
loss = MSE(y, y_hat, name='loss')

# 优化器
optimizer = GradientDescent(loss, batch_size=batch_size)

# 训练
epoch = 10
for i in range(epoch):
    for j, (X, label) in enumerate(train_data_iter):
        X = X.numpy()
        label = label.numpy()
        losses = []
        for k, x_ in enumerate(X):
            x.set_value(np.mat(x_).T)
            y.set_value(np.mat(label[k]))

            optimizer.one_step()
            loss.forward()
            losses.append(loss.value[0, 0])

        print("epoch:{}, batch:{}, loss:{}".format(i, j, np.mean(losses)))
