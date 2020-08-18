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
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
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

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / data_len

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(dataset, model, epochs):

    # TODO lr 인자로 받기
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)    
    TRG_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    
    if not os.path.exists(f'checkpoints/'):
        os.mkdir(f'checkpoints/')
    if not os.path.exists(f'checkpoints/{TIMESTAMP}'):
        os.mkdir(f'checkpoints/{TIMESTAMP}')
    
    best_valid_loss = float('inf')
    train_iter, val_iter, _ = dataset.build_iterator()

    file = f'checkpoints/{TIMESTAMP}/{TIMESTAMP}_loss.log'
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
                    f'checkpoints/{TIMESTAMP}/model_best.pt')
                else:
                    torch.save(model.state_dict(), 
                    f'checkpoints/{TIMESTAMP}/model_best.pt')
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
def test(dataset, model):
    model.load_state_dict(torch.load('checkpoints/2020-08-17T10:04:51/model_best.pt'))
    _, _, test_iterator = dataset.build_iterator()
    TRG_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    test_loss = evaluate_epoch(model, test_iterator, criterion)
    
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
