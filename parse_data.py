from torch.utils.data import Dataset
import torch.nn as nn
import torch
from parse_embedding import *
import numpy as np


only_one_label = True
data_row = 400
train_num = 2342  # 2342
test_num = 2228  # 2228
vectors, words, word2idx, _ = load()
embed = nn.Embedding(vocab_size, embedding_dim)
embed.weight = nn.Parameter(torch.FloatTensor(vectors))


class MyDataset(Dataset):
    def __init__(self, fileName, num):
        data_tensor = np.zeros((num, data_row, 300), dtype=np.float32)
        target_tensor = np.zeros((num, ), dtype=np.int64)
        fp = open(fileName, encoding='utf-8')
        for idx, l in enumerate(fp):
            if idx == num:
                break
            line = l.split()
            label = [int(s[3:]) for s in line[2:10]]
            if only_one_label:
                M = max(label)
                for i in range(8):
                    if label[i] == M:
                        target_tensor[idx] = i
            else:
                total = sum(label)
                label = [c / total for c in label]
                target_tensor[idx, :] = label
            d = [word2idx[word] for word in line[10:] if word in words]
            if len(d) < data_row:
                d.extend([vocab_size - 1] * (data_row - len(d)))
            elif len(d) > data_row:
                d = d[:data_row]
            data = embed(torch.LongTensor(d))
            data_tensor[idx, :, :] = data.detach().numpy()
            if idx % 10 == 0:
                print('preparing dataset %s, %d / %d' % (fileName, idx, num))
        self.data_tensor = torch.from_numpy(data_tensor)
        self.target_tensor = torch.from_numpy(target_tensor)

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


if __name__ == '__main__':
    train = MyDataset('sina/sinanews.train', train_num)
    torch.save(train, 'data/train.pt')
    test = MyDataset('sina/sinanews.test', test_num)
    torch.save(test, 'data/test.pt')


