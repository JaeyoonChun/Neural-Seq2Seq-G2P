import torch
import torch.nn as nn
import numpy as np
import random
import os

from Train import train
from Test import test
from biLSTM_attn import Encoder, Attention, Decoder, Seq2Seq
from data_loader import Librispeech, DataLoader

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def build_model(dataset, device):
    INPUT_DIM = len(dataset.G_FIELD.vocab)
    OUTPUT_DIM = len(dataset.P_FIELD.vocab)
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, 1, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attn, 1, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    return model

def main(fpath, batch_size, mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DataLoader(fpath, Librispeech, batch_size, device)
    model = build_model(dataset, device).to(device)

    if mode == 'train':
        train(dataset, model, 30)
    if mode == 'test':
        test(dataset, model, device)

if __name__ == '__main__':
    main('data/', 16, 'test')