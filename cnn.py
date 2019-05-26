import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        C = args.label_num
        channelIn = 1
        kernelNum = args.kernel_num
        kernelSizes = args.kernel_sizes
        embeddingDim = args.embed_dim

        self.convs1 = nn.ModuleList([nn.Conv2d(channelIn, kernelNum, (kernelSize, embeddingDim)) for kernelSize in kernelSizes])
        for conv in self.convs1:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0.1)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(kernelSizes) * kernelNum, C)

    def forward(self, x):
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        # logit = F.softmax(x)
        return logit





