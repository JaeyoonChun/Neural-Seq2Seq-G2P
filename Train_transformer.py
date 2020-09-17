import torch
import torch.optim as optim
import torch.nn as nn

import math
from functools import partial
from tqdm import trange
import datetime
import time
import os

TIMESTAMP = datetime.datetime.now().isoformat()[:19]
print(TIMESTAMP)

def train_epoch(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    data_len= len(iterator)
    pbar = trange(data_len)
    iterator = iter(iterator)

    for i in pbar:
        
        batch = next(iterator)
        src = batch.grapheme
        trg = batch.phoneme
        
        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])
        # TODO 찾아보기
        # 왜 biLSIM 에서는 eos를 안 자르고 넣고 여기선 자르고 넣음?
        # 왜 biLSTM 에서는 아웃풋에서 sos도 같이 나오고 여기선 안나옴?
        #trg = [trg len, batch size]
        #output = [batch size, trg len-1, output dim]
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        #output = [batch size * (trg len - 1), output dim]
        #trg = [batch size * (trg len - 1)]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / data_len

def evaluate_epoch(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    data_len = len(iterator)
    pbar = trange(data_len)
    iterator = iter(iterator)

    with torch.no_grad():
    
        for i in pbar:
            
            batch = next(iterator)
            src = batch.grapheme
            trg = batch.phoneme

            output, _ = model(src, trg[:,:-1]) #turn off teacher forcing
            #trg = [batch size, trg_len]
            #output = [batch size, trg len-1, output dim]

            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            #trg = [batch size * (trg len - 1)]
            #output = [batch size * (trg len - 1), output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / data_len

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(dataset, model, epochs, model_type):

    # TODO lr 인자로 받기
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)    
    TRG_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    
    best_valid_loss = float('inf')
    train_iter, val_iter, _ = dataset.build_iterator()

    if not os.path.exists(f'checkpoints/'):
        os.mkdir(f'checkpoints/')
    if not os.path.exists(f'checkpoints/{TIMESTAMP}_{model_type}'):
        os.mkdir(f'checkpoints/{TIMESTAMP}_{model_type}')

    file = f'checkpoints/{TIMESTAMP}_{model_type}/{TIMESTAMP}_loss.log'
    with open(file, 'w', encoding='utf-8') as wf:
        early_cnt = 0
        _print = partial(print, file=wf, flush=True)

        for epoch in range(epochs):
            
            start_time = time.time()
            
            train_loss = train_epoch(model, train_iter, optimizer, criterion, 1)
            valid_loss = evaluate_epoch(model, val_iter, criterion)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if epoch >= 2:
                lr_scheduler.step()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(),
                    f'checkpoints/{TIMESTAMP}_{model_type}/model_best.pt')
                else:
                    torch.save(model.state_dict(), 
                    f'checkpoints/{TIMESTAMP}_{model_type}/model_best.pt')
                _print('new best model')
                early_cnt = 0
            else:
                early_cnt += 1

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | current lr : {lr_scheduler.get_lr()}')
            _print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | current lr : {lr_scheduler.get_lr()}')

            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            _print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            _print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            print('-' * 20)
            
            # TODO early_step 인자로 받기
            if early_cnt > 3 - 1:
                print('training session has been early stopped')
                _print('training session has been early stopped')
                break

# TODO
def test_temp(dataset, model, model_path):
    model.load_state_dict(torch.load(f'{model_path}/model_best.pt'))
    _, _, test_iterator = dataset.build_iterator()
    TRG_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    test_loss = evaluate_epoch(model, test_iterator, criterion)
    
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
