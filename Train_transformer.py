import torch
import torch.optim as optim
import torch.nn as nn

import math
from functools import partial
from tqdm import trange
import datetime
import time
import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)

def train_epoch(model, iterator, optimizer, criterion, clip, args):
    
    model.train()
    
    epoch_loss = 0
    
    data_len= len(iterator)
    pbar = trange(data_len)
    iterator = iter(iterator)
    step = 0
    for i in pbar:
        
        batch = next(iterator)
        src = batch.grapheme
        trg = batch.phoneme
        #trg = [trg len, batch size]

        optimizer.zero_grad()
        if args.LSTM:
            output = model(src, trg)
            output_dim = output.shape[-1]
            #output = [trg len, batch size, output dim]
                    
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

        elif args.Transformer:
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
        
        #output = [(trg len - 1) * batch size, output dim]
        #trg = [(trg len - 1) * batch size]
    
        loss = criterion(output, trg)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        epoch_loss += loss.item()
        if (i + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            optimizer.step()
            step += 1
        
 
        
    return epoch_loss / data_len, step

def evaluate_epoch(model, iterator, criterion, args):
    
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

            if args.LSTM:
                output = model(src, trg, 0)
                output_dim = output.shape[-1]
                #output = [trg len, batch size, output dim]
                        
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

            elif args.Transformer:
                output, _ = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / data_len

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(dataset, model, args):

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)    
    TRG_PAD_IDX = dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    best_valid_loss = float('inf')
    train_iter, val_iter, _, train_data_len, val_data_len = dataset.build_iterator()
    if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_iter) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_iter) // args.gradient_accumulation_steps * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_iter))
    logger.info("  Num train data = %d | Num valid data = %d", train_data_len, val_data_len)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  learning rate = %f", args.learning_rate)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  saving steps = %d", args.save_steps)
    logger.info(f"  pretrain vector = {args.pretrain_vector}")
    t_steps = 0
    training_stats = []

    for epoch in range(args.num_train_epochs):
        
        start_time = time.time()
        
        train_loss, steps = train_epoch(model, train_iter, optimizer, criterion, args.max_grad_norm, args)
        valid_loss = evaluate_epoch(model, val_iter, criterion)
        t_steps += steps
        training_stats.append(
            {'epoch': epoch + 1,
            'training_loss': train_loss,
            'valid_loss': valid_loss,
            'steps': t_steps
            }
        )
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if epoch >= 2:
            lr_scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(args, model, optimizer, best_valid_loss, epoch)
            early_cnt = 0
        else:
            early_cnt += 1

        logger.info(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | current lr : {lr_scheduler.get_lr()}')
        logger.info(f"  total_steps = {t_steps}")
        logger.info(f"  train_loss = {train_loss:.4f} | 'Train_PPL' = {math.exp(train_loss):.3f} ")
        logger.info(f"  valid_loss = {valid_loss:.4f} | 'Val_PPL' = {math.exp(valid_loss):.3f}")
    
        if early_cnt > args.early_cnt - 1:
            logger.info('training session has been early stopped')
            df_stats = pd.DataFrame(data=training_stats, )
            df_stats = df_stats.set_index('epoch')
            df_stats.to_csv(f'./checkpoints/{args.version}/stats.csv', sep='\t', index=True)
            break
def save_model(args, model, optimizer, loss, epoch):
    fpath = f'./checkpoints/{args.version}/'
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    torch.save({
        'epoch': epoch,
        'model_stat_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, os.path.join(fpath, 'model.pt'))
    torch.save(args, os.path.join(fpath, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", fpath)
