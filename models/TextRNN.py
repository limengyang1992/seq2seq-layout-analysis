# coding: UTF-8
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.rnn_n_vocab,
                                      config.rnn_embed,
                                      padding_idx=config.rnn_n_vocab-1)

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(config.rnn_embed,
                            config.rnn_hidden_size,
                            1,    # num_layers
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Linear(config.rnn_hidden_size, len(config.class_idx))

    def forward(self, x):
        x = x.permute(1, 0)
        emb = self.embedding(x)
        emb = self.dropout(emb)
        hidden, _ = self.lstm(emb)  # 句子最后时刻的 hidden state
        hidden = hidden[:, -1, :]
        out = self.fc(hidden)
        return out, hidden
