import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args

        C = args.label_num
        embeddingDim = args.embed_dim
        sz = [400 * embeddingDim, 4096, 512, 64, C]

        self.fc = nn.ModuleList([nn.Linear(sz[i], sz[i + 1]) for i in range(len(sz) - 1)])
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        if self.args.static:
            x = Variable(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        for fc in self.fc:
            x = fc(x)
        logit = F.softmax(x)
        return logit





