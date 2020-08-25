import torch
import torch.nn as nn
import numpy as np
import random
import os

from Train_transformer import train, test_temp
from Test_transformer import test
from transformer import Encoder, Decoder, Seq2Seq
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
    HID_DIM = 256
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    G_PAD_IDX = dataset.G_FIELD.vocab.stoi[dataset.G_FIELD.pad_token]
    P_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]
    model = Seq2Seq(enc, dec, G_PAD_IDX, P_PAD_IDX, device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    return model

def main(fpath, batch_size, mode, model_type, model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DataLoader(fpath, Librispeech, batch_size, device, model_type)
    model = build_model(dataset, device).to(device)

    if mode == 'train':
        train(dataset, model, 100, model_type)
    if mode == 'test':
        test(dataset, model, device, model_path)
    if mode == 't':
        test_temp(dataset, model, model_path)


if __name__ == '__main__':
    model_path = 'checkpoints/2020-08-25T19:18:53_transformer'
    main('data/', 32, 'train', 'transformer', model_path)