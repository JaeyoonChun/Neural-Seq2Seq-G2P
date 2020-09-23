import torchtext.data as data
from torchtext.data import Field, BucketIterator, Iterator
from utils import get_word_file
import os
import re
import json

from transformers import AutoTokenizer
from torchtext.vocab import GloVe, FastText

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
    def splits(cls, fpath, g_field, p_field, do_lr=False):
        # with open(fpath+'word_train.txt', 'r', encoding='utf-8') as train_f,\
        # open(fpath+'word_dev.txt', 'r', encoding='utf-8') as val_f,\
        # open(fpath+'word_test.txt', 'r', encoding='utf-8') as test_f:
        #     train_lines = train_f.readlines()
        #     val_lines = val_f.readlines()
        #     test_lines = test_f.readlines()

        with open(fpath+'librispeech_train-clean.json', 'r', encoding='utf-8') as train_f,\
        open(fpath+'librispeech_dev-clean.json', 'r', encoding='utf-8') as val_f,\
        open(fpath+'librispeech_test-clean.json', 'r', encoding='utf-8') as test_f:
            train_data = json.load(train_f)
            val_data = json.load(val_f)
            test_data = json.load(test_f)
        if do_lr:
            train_data = cls(train_data[:500], g_field, p_field)
        else:
            train_data = cls(train_data, g_field, p_field)
        val_data = cls(val_data, g_field, p_field)
        test_data = cls(test_data, g_field, p_field)

        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(val_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")

        return (train_data, val_data, test_data)
    
class DataLoader:
    def __init__(self, fpath, librispeech, batch_size, device, model_type, args):
        
        self.batch_first = False
        if model_type == 'transformer':
            self.batch_first = True
        
        # self.pat_space = re.compile(r'(?<=[A-Z0-9])\s(?=[A-Z0-9])')
        # self.pat_replace_space = re.compile('#')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.G_FIELD = Field(init_token='<sos>',
                eos_token='<eos>',
                tokenize=lambda x:x.split(),
                batch_first=True)
        self.P_FIELD = Field(init_token='<sos>',
                eos_token='<eos>',
                tokenize=self.phoneme_tokenizer,
                batch_first=True)
        # if not os.path.exists(os.path.join(fpath, 'word_train.txt')):
        #     self.load_dataset(fpath)
        self.librispeech = librispeech
        self.train_data, self.val_data, self.test_data = self.librispeech.splits(fpath, self.G_FIELD, self.P_FIELD, args.do_lr)
        self.batch_size = batch_size
        self.device = device
        print(vars(self.train_data.examples[20]))
        print(vars(self.val_data.examples[20]))

        # self.G_FIELD.build_vocab(self.train_data, self.val_data, self.test_data)
        self.G_FIELD.build_vocab(self.train_data, self.val_data, self.test_data, vectors=GloVe(name='6B', dim=300))
        # self.G_FIELD.build_vocab(self.train_data, self.val_data, self.test_data, vectors=FastText(language='en'))

        self.P_FIELD.build_vocab(self.train_data, self.val_data, self.test_data)
        print(len(self.G_FIELD.vocab))
        print(len(self.P_FIELD.vocab))
        # print(self.G_FIELD.vocab.stoi)
        # print(self.G_FIELD.vocab.vectors[0])
        # print(self.G_FIELD.vocab.vectors[10])
        # raise Exception

    def load_dataset(self, fpath):
        for type in 'train dev test'.split():
            get_word_file(fpath, type)
    
    def phoneme_tokenizer(self, pho):
        # return pho.split(' _ ')
        return pho.split()
        # pho = self.pat_space.sub('#', pho).split()
        # return [self.pat_replace_space.sub(' ', w) for w in pho]

    def build_iterator(self):
        train_iter = Iterator(self.train_data, batch_size=self.batch_size, train=True, device=self.device, shuffle=True)
        val_iter = Iterator(self.val_data, batch_size=self.batch_size, train=False, device=self.device, shuffle=False)
        test_iter = Iterator(self.test_data, batch_size=1, train=False, device=self.device)

        # batch = next(iter(train_iter))
        # print(batch.grapheme)
        # print(batch.grapheme.shape)
        # print(batch.phoneme)
        # print(batch.phoneme.shape)
        # print(' '.join([self.G_FIELD.vocab.itos[i] for i in batch.grapheme.view(-1).detach().cpu().numpy()]))
        # print(' '.join([self.P_FIELD.vocab.itos[i] for i in batch.phoneme.view(-1).detach().cpu().numpy()]))       

        # batch = next(iter(val_iter))
        # print(batch.grapheme)
        # print(batch.grapheme.shape)
        # print(batch.phoneme)
        # print(batch.phoneme.shape)
        # print(' '.join([self.G_FIELD.vocab.itos[i] for i in batch.grapheme.view(-1).detach().cpu().numpy()]))
        # print(' '.join([self.P_FIELD.vocab.itos[i] for i in batch.phoneme.view(-1).detach().cpu().numpy()]))     
        # raise Exception('dd')
        return (train_iter, val_iter, test_iter, len(self.train_data), len(self.val_data))

if __name__ == '__main__':
    dataset = DataLoader('./data/', Librispeech, 32, 'cpu', 'transformer')
    dataset.build_iterator()