
import random
import json
import numpy as np
import pandas as pd

import torch
from torchtext import data, vocab
from torch.utils.data import DataLoader,Dataset
import glob


class RnnDataset(data.Dataset):
    def __init__(self, csv_path, text_field, label_field, aug=False, **kwargs):

        csv_data = pd.read_csv(csv_path)
        # 数据处理操作格式
        fields = [("text", text_field), ("label", label_field)]
        examples = []
        for text, label in zip(csv_data['text'], csv_data['label']):
            examples.append(data.Example.fromlist([str(text), label], fields))

        super(RnnDataset, self).__init__(examples, fields)


def rnn_iter(train_path, test_path, batchsize, TEXT, LABEL):
    train = RnnDataset(train_path, text_field=TEXT,
                       label_field=LABEL,  aug=1)
    test = RnnDataset(test_path, text_field=TEXT,
                      label_field=None, aug=1)
    # 传入用于构建词表的数据集
    vectors = vocab.Vectors(name="wordvec.txt", cache="data")
    TEXT.build_vocab(test, vectors=vectors)
    weight_matrix = TEXT.vocab.vectors
    # 同时对训练集和验证集构造迭代器
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test),
        batch_sizes=(batchsize, batchsize),
        device=torch.device('cuda'),
        sort_key=lambda x: len(x.text),
        sort_within_batch=False)

    return train_iter, test_iter, weight_matrix


def stoi(string, max_len, dicts):
    # <unk>=0 <pad>=1
    result = [1]*max_len
    length = len(string)
    convert = []
    for i in string:
        if i in dicts.keys():
            convert.append(dicts[i])
        else:
            convert.append(0)
    result[:length] = convert[:max_len]
    return np.array(result)

