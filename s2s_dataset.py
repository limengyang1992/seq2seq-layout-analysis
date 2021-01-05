import glob
import random
import json
import numpy as np
import pandas as pd

import torch
from torchtext import data, vocab
from torch.utils.data import DataLoader, Dataset

import rnn_infer
from config import Config

config = Config()


def str_count(string):
    '''找出字符串中的中英文、空格、数字、标点符号个数'''
    count = {}
    count_en = count_dg = count_sp = count_zh = count_pu = 0
    try:
        for s in string:
            # 英文
            if s >= 'a' and s <= 'z' or s >= 'A' and s <= 'Z':
                count_en += 1
            # 数字
            elif s.isdigit():
                count_dg += 1
            # 空格
            elif s.isspace():
                count_sp += 1
            # 中文
            elif s.isalpha():
                count_zh += 1
            # 特殊字符
            else:
                count_pu += 1
    except:
        pass

    count["EN"] = count_en
    count["DG"] = count_dg
    count["SP"] = count_sp
    count["ZH"] = count_zh
    count["PU"] = count_pu
    return count



def ifdrop(string):
    '''
    根据业务情况过滤无用文本框
    '''
    if len(string)<2:
        return False
    
    count = str_count(string)
    if count["ZH"]>6:
        return False
    
    if count["EN"] == len(string):
        return False
    return True



def change_num(string):
    '''
    根据业务需求,随即替换数字
    '''
    ss = list(string)
    for i in range(len(ss)):
        if ss[i].isdigit():
            n = random.randint(0, 9)
            ss[i] = str(n)
    return ''.join(ss)


def processing_location(loc_list, h, w):
    loc_hidden = []
    for points in loc_list:
        if len(points) == 2:
            p2 = [points[1][0], points[0][1]]
            p3 = [points[0][0], points[1][1]]
            location = points[0]+p2+points[1]+p3
        else:
            location = points[0]+points[1]+points[2]+points[3]

        noise = [random.randint(-3, 3) for _ in range(8)]
        add_noise = list(np.array(location)+np.array(noise))
        add_noise = [x/w if i % 2 == 0 else x /
                     h for i, x in enumerate(add_noise)]
        flatten = list(add_noise)*config.expend_box_times
        loc_hidden.append(flatten)
    return np.array(loc_hidden).T


def processing_text(label_list):
    out = rnn_infer.predict(label_list)
    return out


def json2embed(path, max_len):

    output = np.zeros((config.s2s_emb_dim, max_len))
    text_list = []
    loc_list = []
    label_list = []

    with open(path, 'rb') as f:
        data = json.loads(f.read())

    h = data["imageHeight"]
    w = data["imageWidth"]
    shapes = data["shapes"]
    shapes = sorted(shapes, key=lambda x: (
        x["points"][0][1], x["points"][0][1]))
        
    for i, obj in enumerate(shapes):

        label = str(obj["label"]).replace("\"", "").replace("\'", "")
        if ifdrop(label):
            group_id = obj["group_id"]
            points = obj["points"]

            if len(points) in [2, 4]:
                text_list.append(label)
                loc_list.append(points)
                if group_id:
                    label_list.append(int(group_id))
                else:
                    label_list.append(0)

    length = len(text_list) if len(text_list) < max_len+1 else max_len

    out_label = [0]*max_len 
    out_label[:length] = label_list

    text_hidden = processing_text(text_list[:max_len])
    loc_hidden = processing_location(loc_list, h, w)

    output[:config.rnn_hidden_size, :length] = text_hidden[:, :length]
    output[config.rnn_hidden_size:, :length] = loc_hidden[:, :length]
    return output.T, out_label


def gen(paths, batch_size, seq_length, device):
    num =len(paths)
    i=0
    while True:
        X = np.zeros((batch_size, seq_length, config.s2s_emb_dim))*1.0
        Y = np.zeros((batch_size, seq_length))
        for j in range(batch_size):
            if i>=num:
                i=0
                np.random.shuffle(paths)
            path = paths[i]
            i+=1
            a,b = json2embed(path, seq_length)
            X[j], Y[j] = a,b
        outx = torch.from_numpy(X).float().to(device)
        outy = torch.from_numpy(Y).to(device)      
        yield  outx, outy  
            



if __name__ == "__main__":

    import pandas as pd
    paths = glob.glob("data/train/*.json")
    device = torch.device('cuda')
    train_loader = gen(paths, config.s2s_batch_size, config.max_box_num, device)
    for i in train_loader:
        print(i[0])

            
