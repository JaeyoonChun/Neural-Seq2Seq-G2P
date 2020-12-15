import torch
import torch.nn as nn
import numpy as np
import random
import os

from Train_transformer import train
from Test_transformer import test
from transformer import Encoder, Decoder, G2P
from data_loader import Librispeech, DataLoader
from utils import init_logger, set_seeds
import argparse

def build_model(dataset, device):
    INPUT_DIM = len(dataset.G_FIELD.vocab)
    OUTPUT_DIM = len(dataset.P_FIELD.vocab)
    HID_DIM = 300
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 10
    DEC_HEADS = 10
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, vectors=dataset.G_FIELD.vocab.vectors)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    G_PAD_IDX = dataset.G_FIELD.vocab.stoi[dataset.G_FIELD.pad_token]
    P_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]
    model = G2P(enc, dec, G_PAD_IDX, P_PAD_IDX, device)

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)
    model.apply(initialize_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    return model

def main(args):
    init_logger(args)
    set_seeds()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DataLoader(args.data_dir, Librispeech, args.train_batch_size, device, 'transformer', args)
    model = build_model(dataset, device).to(device)
    
    if args.do_train:
        train(dataset, model, args)
    if args.do_test:
        test(dataset, model, device, f'checkpoints/{args.version}')
  


if __name__ == '__main__':
    # model_path = 'checkpoints/2020-09-22T20:29:48_transformer'
    # main('data/', 64, 'train', 'transformer', model_path)


    parser = argparse.ArgumentParser()

    parser.add_argument("--test_model_dir", default="./models/third", type=str, help="Path load testing model")
    parser.add_argument("--save_model_dir", default="./checkpoints/", type=str, help="Path to save trained model")

    parser.add_argument("--data_dir", default="./data/", type=str, help="Train file")
    parser.add_argument("--train_data_dir", default="librispeech_train-clean.json", type=str, help="Train file")
    parser.add_argument("--valid_data_dir", default="librispeech_dev-clean.json", type=str, help="valid file")
    parser.add_argument("--test_data_dir", default="librispeech_test-clean.json", type=str, help="Test file")

    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")

    parser.add_argument("--learning_rate", default=0.00079, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=500, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--logging_steps', type=int, default=1, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="")
    parser.add_argument('--max_steps', type=int, default=0, help="Max steps to train")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Max sequence length.")
    
    parser.add_argument("--do_train",  action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test",  action="store_true", help="Whether to run training.")
    parser.add_argument("--do_lr",  action="store_true", default=False, help="Whether to run training.")

    parser.add_argument("--early_cnt",  type=int, default=3, help="Whether to run training.")
    parser.add_argument("--version", type=str, default='trained_0.00079', help="Train a sentiment classifier for the model 4")
    parser.add_argument("--pretrain_vector", type=str, default='GloVe', help="Train a sentiment classifier for the model 4")
    parser.add_argument("--cuda_num", type=str, default='0', help="Set CUDA_VISIBLE_DEVICES to use GPU device.")

    args = parser.parse_args()

    main(args)

