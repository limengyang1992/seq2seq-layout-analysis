import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
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
    model.load_state_dict(torch.load('weight/s2s.pt'))
    model.eval()

    data = gen(["data/test/96.json"], 1, config.max_box_num, device)
    
    with torch.no_grad():
        src, trg  = next(data)
        output = model(src)
        output = output.permute(1, 0, 2).contiguous().view(-1, len(config.class_char))
        output = torch.max(F.softmax(output,dim=1),1)
        possible,label= output.values,output.indices
        acc = np.mean((label == trg).cpu().numpy())

        print("trget:",trg.long())
        print("label:",label)
        print("possible:",possible)
        print("acc:",acc)
        
        
