import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, model_name, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.model_name = model_name
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        nn.init.xavier_normal_(self.lstm.all_weights[0][0])
        nn.init.xavier_normal_(self.lstm.all_weights[0][1])
        nn.init.xavier_normal_(self.lstm.all_weights[1][0])
        nn.init.xavier_normal_(self.lstm.all_weights[1][1])
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        nn.init.xavier_normal_(self.gru.all_weights[0][0])
        nn.init.xavier_normal_(self.gru.all_weights[0][1])
        nn.init.xavier_normal_(self.gru.all_weights[1][0])
        nn.init.xavier_normal_(self.gru.all_weights[1][1])
        self.fc = nn.Linear(hidden_dim * n_layers * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        x = self.dropout(text)
        if self.model_name == 'LSTM':
            output, (hidden, cell) = self.lstm(x)
        elif self.model_name == 'GRU':
            output, hidden = self.gru(x)

        # hidden = [batch size, hidden dim * num directions]
        hidden = self.dropout(torch.cat([hidden[i, :, :] for i in range(hidden.shape[0])], dim=1))

        return F.softmax(self.fc(hidden), dim=1)
