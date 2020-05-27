from sklearn.datasets import load_boston
from torch.utils.data import DataLoader, Dataset, random_split

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
train_data_iter = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last = True)


