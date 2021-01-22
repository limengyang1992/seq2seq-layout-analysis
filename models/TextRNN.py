# coding: UTF-8
import torch
import torch.nn as nn


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        H0 = self.tanh(x)
        H1 = torch.matmul(H0, self.w)
        H2 = nn.functional.softmax(H1, dim=1)
        alpha = H2.unsqueeze(-1)
        att_hidden = torch.sum(x * alpha, 1)
        return att_hidden, H2


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

        self.attention = Attention(config.rnn_hidden_size)
        
        self.fc = nn.Linear(config.rnn_hidden_size, len(config.class_idx))

    def forward(self, x):
        x = x.permute(1, 0)
        emb = self.embedding(x)
        emb = self.dropout(emb)
        hidden, _ = self.lstm(emb)  
        attn, alpha = self.attention(hidden)
        out = self.fc(attn)
        return out, hidden
