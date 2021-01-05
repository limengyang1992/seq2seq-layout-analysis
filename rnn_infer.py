# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np

from models import TextRNN
from config import Config


config = Config()
# infer
embed = np.load(config.embed_path)
dicts = np.load(config.stoi_path, allow_pickle=True).item()
config.rnn_n_vocab = len(embed)
# config.rnn_n_vocab = len(weight_matrix)
model = TextRNN.Model(config).to(config.device)
model.load_state_dict(torch.load(config.rnn_path))
model.eval()


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
    return result


def predict(strings):
    inputs = []
    for string in strings:
        ids = stoi(string, config.max_text_len, dicts)
        if len(ids) != config.max_text_len:
            print(string)
        inputs = inputs+ids
    inputs_array = np.array(inputs).reshape(-1, config.max_text_len)
    with torch.no_grad():
        input_tensor = torch.from_numpy(inputs_array).t()
        input_tensor = input_tensor.to(config.device)
        types, hidden = model(input_tensor)
        output = hidden.cpu().numpy().T
    return output


if __name__ == "__main__":


    print(predict(["冀国税票证服务中心2015年1月印2.1万卷",
                #    "市银有限公号",
                   "2.10",
                #    "o",
                #    "代码",
                #    "等候:",
                #    "00000000",
                #    "10.00",
                #    "17:16",
                #    "四川汇利",
                   "00513465"]))
