"""
周杰伦歌词数据集处理
"""

import random
import zipfile
import numpy as np

jaychou_lyrics_path = r"../data/jaychou_lyrics.txt.zip"
jaychou_lyrics_file = "jaychou_lyrics.txt"


def load_data_jay_lyrics():
    """ 加载歌词 """
    with zipfile.ZipFile(jaychou_lyrics_path) as zin:
        with zin.open(jaychou_lyrics_file) as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps):
    """ 随机采样 """
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X), np.array(Y)


def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    """ 相邻采样 """
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    num_steps, batch_size = 35, 32

    for X, Y in data_iter_consecutive(corpus_indices, batch_size, num_steps):
        print(X.shape)
        print(X)
        print('*' * 100)
        print(Y.shape)
        print(Y)
        print('*' * 100)
        y = Y.T.reshape((-1,))
        print(y.shape)
        print(y)
        break
