import torchtext.data as data
from torchtext.data import Field, BucketIterator
import os
import re
import json

from transformers import AutoTokenizer
import logging
import pickle
logger = logging.getLogger(__name__)

def build_dataset(opt, args, device, vectors):
    batch_first = False
    if opt.model_type == 'Transformer':
        batch_first = True
        
    if args.word_train:
        G_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        P_tokenizer = lambda x: x.split(' _ ')
    if args.token_train:
        G_tokenizer = lambda x: list(x)
        P_tokenizer = lambda x: x.split()

    G_FIELD = Field(init_token='<sos>',
            eos_token='<eos>',
            tokenize=G_tokenizer,
            batch_first=batch_first)
    P_FIELD = Field(init_token='<sos>',
            eos_token='<eos>',
            tokenize=P_tokenizer,
            batch_first=batch_first)
    fields = dict([('grapheme', G_FIELD), ('phoneme', P_FIELD)])
    train_data, val_data, test_data = Librispeech.splits(args.data_dir, args.data_type, G_FIELD, P_FIELD)
    
    if vectors:
        from torchtext.vocab import GloVe
        G_FIELD.build_vocab(train_data, val_data, test_data, vectors=GloVe(name='6B', dim=300))
    else:
        G_FIELD.build_vocab(train_data, val_data, test_data)
    P_FIELD.build_vocab(train_data, val_data, test_data)
    
    logger.info(vars(train_data.examples[0]))
    logger.info(vars(val_data.examples[0]))

    train_iter = BucketIterator(train_data, batch_size=args.train_batch_size, train=True, device=device, shuffle=True)
    val_iter = BucketIterator(val_data, batch_size=args.eval_batch_size, train=False, device=device, shuffle=False)
    test_iter = BucketIterator(test_data, batch_size=args.eval_batch_size, train=False, device=device)

    return (train_iter, val_iter, test_iter), fields


class Librispeech(data.Dataset):    
    def __init__(self, data_lines, G_FIELD, P_FIELD):
        fields = [('grapheme', G_FIELD), ('phoneme', P_FIELD)]
        examples = []
        for line in data_lines:
            grapheme = line['G']
            phoneme = line['P']
            examples.append(data.Example.fromlist([grapheme, phoneme], fields))
        self.sort_key = lambda x:len(x.grapheme)
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, fpath, data_type, G_FIELD, P_FIELD):
        with open(fpath+f'/{data_type}_train.json', 'r', encoding='utf-8') as train_f,\
        open(fpath+f'/{data_type}_dev.json', 'r', encoding='utf-8') as val_f,\
        open(fpath+f'/{data_type}_test.json', 'r', encoding='utf-8') as test_f:
            train_data = json.load(train_f)
            val_data = json.load(val_f)
            test_data = json.load(test_f)
      
        train_dataset = cls(train_data, G_FIELD, P_FIELD)
        val_dataset = cls(val_data, G_FIELD, P_FIELD)
        test_dataset = cls(test_data, G_FIELD, P_FIELD)

        logger.info(f"Number of training examples: {len(train_dataset.examples)}")
        logger.info(f"Number of validation examples: {len(val_dataset.examples)}")
        logger.info(f"Number of testing examples: {len(test_dataset.examples)}")

        return (train_dataset, val_dataset, test_dataset)