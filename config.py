import numpy as np
import torch


class Config(object):

    """配置参数"""

    def __init__(self):
        # share
        self.device = torch.device('cuda')                              # 设备

        self.train_jsons = "data/train"                                 # 训练集（labelme格式）
        self.test_jsons = "data/test"                                   # 测试集（labelme格式）
        self.rnn_train_path = "data/train.csv"
        self.rnn_test_path = "data/test.csv"
        self.w2v_path = "data/wordvec.txt"
        self.embed_path = "weight/embed.npy"
        self.stoi_path = "weight/stoi.npy"
        self.rnn_path = "weight/rnn.pt"
        self.s2s_path = "weight/s2s.pt"

        self.class_char = ["其他", "代码", "号码", "车牌号", "日期",
                           "上车",  "下车", "单价", "里程", "金额"]        # 标签列表（对应groupid）

        self.num_classes = len(self.class_char)
        self.class_idx = [x for x in range(self.num_classes)]
        self.max_text_len = 20                                          # 最大文本长度
        self.max_box_num = 50                                           # 最大box个数
        self.expend_box_times = 8                                       # box扩增倍数

        # rnn
        self.rnn_hidden_size = 64                                       # 句向量维度
        self.rnn_num_epochs = 100
        self.rnn_batch_size = 64
        self.rnn_embed = 100
        self.rnn_dropout = 0.5
        self.rnn_n_vocab = 0
        self.rnn_learning_rate = 1e-3

        # seq2seq
        self.s2s_epoch = 100
        self.s2s_batch_size = 16

        self.s2s_clip = 1
        self.s2s_enc_hid = 128
        self.s2s_dec_hid = 128
        self.s2s_enc_dropout = 0.5
        self.s2s_dec_dropout = 0.5
        self.s2s_emb_dim = self.rnn_hidden_size + self.expend_box_times*8
