from torch.utils.data import Dataset
import torch.nn as nn
import torch
import numpy as np
from random import choice
from gensim.models import KeyedVectors

only_one_label = True
data_row = 400
num = [2342, 2228]
model = KeyedVectors.load_word2vec_format('data/model')

class MyDataset(Dataset):
    def __init__(self, arg):
        if arg == 'train':
            data_tensor = np.zeros((num[0], 400, 80), dtype=np.float32)
            target_tensor = np.zeros((num[0], ), dtype=np.int64)
            In = open('sina/train.in', encoding='utf-8')
            out = open('sina/train.out', encoding='utf-8')
        else:
            data_tensor = np.zeros((num[1], 400, 80), dtype=np.float32)
            target_tensor = np.zeros((num[1], ), dtype=np.int64)
            In = open('sina/test.in', encoding='utf-8')
            out = open('sina/test.out', encoding='utf-8')
        for idx, l in enumerate(out):
            line = list(map(int, l.split()))
            m = max(line[1:])
            tmp = []
            for i in range(1, 9):
                if line[i] == m:
                    tmp.append(i - 1)
            target_tensor[idx] = choice(tmp)
        for idx, l in enumerate(In):
            for i, word in enumerate(l.split()):
                if i >= 400:
                    break
                data_tensor[idx, i, :] = model[word]
            print(idx)
        self.data_tensor = torch.from_numpy(data_tensor)
        self.target_tensor = torch.from_numpy(target_tensor)

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

if __name__ == '__main__':
    train = MyDataset('train')
    torch.save(train, 'data/train.data')
    test = MyDataset('test')
    torch.save(test, 'data/test.data')
    # model = KeyedVectors.load_word2vec_format('data/model')
    # while 1:
    #     print(model[input()])


