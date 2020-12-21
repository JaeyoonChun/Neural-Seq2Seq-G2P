from attrdict import AttrDict
import json
import argparse
import os

from G2P.modules.utils import set_seeds, load_device, init_logger
from G2P.build_model import build_model
from G2P.data_loader import build_dataset
from G2P.Train import Trainer


def main(opt):
    model_args_path = os.path.join('G2P/config', opt.model_type+'.json')
    train_args_path = os.path.join('G2P/config', 'train_config.json')
    with open(model_args_path, 'r', encoding='utf-8') as f:
        model_args = AttrDict(json.load(f))
    with open(train_args_path, 'r', encoding='utf-8') as f:
        train_args = AttrDict(json.load(f))
    
    if train_args.word_train:
        assert model_args.vectors == True
    elif train_args.token_train:
        assert model_args.vectors == False
    
    init_logger()
    set_seeds()
    device = load_device(train_args)

    (train_iter, val_iter, _), fields = build_dataset(opt, train_args, device, model_args.vectors)
    model = build_model(model_args, opt, fields, device)
    model = model.to(device)
    trainer = Trainer(train_args, opt, model, fields)
    trainer.train(train_iter, val_iter)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='LSTM', required=True, type=str)
    opt = parser.parse_args()
    main(opt)

#TODO model_param