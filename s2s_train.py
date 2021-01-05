import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import math
import time
import glob

from config import Config
from s2s_dataset import gen
from torch.utils.data import DataLoader, Dataset
from models.Seq2Seq import Attention, Encoder, Decoder, Seq2Seq


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, steps, iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for i in range(steps):

        src, trg = next(iterator)
        optimizer.zero_grad()
        output = model(src)
        output = output.permute(1, 0, 2).contiguous().view(-1, len(config.class_char))
        trg = trg.contiguous().long().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        acc = np.mean((torch.argmax(output, 1) == trg).cpu().numpy())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / steps, epoch_acc / steps


def evaluate(model, steps, iterator, criterion):

    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    for i in range(steps):

        src, trg = next(iterator)
        output = model(src)  
        output = output.permute(1, 0, 2).contiguous().view(-1, len(config.class_char))
        trg = trg.contiguous().long().view(-1)
        loss = criterion(output, trg)
        acc = np.mean((torch.argmax(output, 1) == trg).cpu().numpy())
        # print(torch.argmax(output,1))
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / steps, epoch_acc / steps


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    config = Config()
    device = config.device

    attn = Attention(config.s2s_enc_hid, config.s2s_dec_hid)
    enc = Encoder(config.s2s_emb_dim,
                  config.s2s_enc_hid,
                  config.s2s_dec_hid,
                  config.s2s_enc_dropout)
    dec = Decoder(len(config.class_char),
                  config.s2s_emb_dim,
                  config.s2s_enc_hid,
                  config.s2s_dec_hid,
                  config.s2s_enc_dropout,
                  attn)

    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} train parameters')

    optimizer = optim.Adam(model.parameters())

    # weight_CE = torch.FloatTensor([0.1,1,1,1,1,1,1,1,1,1]).to(device)
    # criterion = nn.CrossEntropyLoss(weight = weight_CE)
    criterion = nn.CrossEntropyLoss()

    # train data
    train_paths = glob.glob("data/train/*.json")
    test_paths = glob.glob("data/test/*.json")
    train_steps = len(train_paths)//config.s2s_batch_size+1
    test_steps = len(test_paths)//config.s2s_batch_size+1
    train_iterator = gen(train_paths, config.s2s_batch_size, config.max_box_num, device)
    test_iterator = gen(test_paths, config.s2s_batch_size, config.max_box_num, device)


    best_valid_loss = float('inf')

    for epoch in range(config.s2s_epoch):

        start_time = time.time()
        train_loss, train_acc = train(model, train_steps, train_iterator,
                                      optimizer, criterion, config.s2s_clip)
        valid_loss, valid_acc = evaluate(
            model, test_steps, test_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config.s2s_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train ACC: {train_acc:.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. ACC: {valid_acc:.3f}')
