# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.rnn_learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(config.rnn_num_epochs):
        print('Epoch [{}/{}]'.format(epoch, config.rnn_num_epochs))
        # scheduler.step() # 学习率衰减
        for i, data in enumerate(train_iter):
            trains, labels = data.text, data.label
            outputs, _ = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 30 == 0:

                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.rnn_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc,
                                 dev_loss, dev_acc, time_dif, improve))

                model.train()
            total_batch += 1

    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.rnn_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, data in enumerate(train_iter):
            texts, labels = data.text, data.label
            outputs, _ = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, target_names=config.class_char, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def concert(ids, t):
    strs = ""
    for i in ids:
        strs += TEXT.vocab.itos[i+t]
    return strs


if __name__ == "__main__":
    from config import Config
    import pandas as pd
    import torchtext.vocab as Vocab
    from torchtext import data, vocab
    from rnn_dataset import rnn_iter
    from models import TextRNN

    config = Config()
    train_path = config.rnn_train_path
    test_path = config.rnn_test_path

    TEXT = data.Field(sequential=True,
                      tokenize=lambda x: [t for t in x],
                      lower=True, fix_length=config.max_text_len)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_iter, test_iter, weight_matrix = rnn_iter(
        train_path, test_path, config.rnn_batch_size, TEXT, LABEL)

    # 存储映射字典、embeding_matrix
    np.save(config.stoi_path, dict(TEXT.vocab.stoi))
    np.save(config.embed_path, weight_matrix)
    # train
    config.rnn_n_vocab = len(weight_matrix)

    model = TextRNN.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, test_iter, test_iter)
    evaluate(config, model, test_iter)
