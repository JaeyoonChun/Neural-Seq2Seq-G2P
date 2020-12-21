import torch
import torch.optim as optim
import torch.nn as nn

import math
import datetime
import time
import os
import logging
import pandas as pd
from fastprogress.fastprogress import master_bar, progress_bar

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, train_args, opt, model, fields):
        self.args = train_args
        self.opt = opt
        self.model = model
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= 0.99) 

        p_field = fields['phoneme']
        TRG_PAD_IDX = p_field.vocab.stoi[p_field.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)   

    def train(self, train_dataloader, valid_dataloader):
        
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
      
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  learning rate = %f", self.args.learning_rate)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  saving steps = %d", self.args.save_steps)
        early_cnt = 0
        global_step = 0
        best_valid_loss = float('inf')
        training_stats = []
        epochs = master_bar(range(self.args.num_train_epochs))
        for epoch in epochs:
            start_time = time.time()
            epoch_loss = 0
            epoch_iterator = progress_bar(train_dataloader, parent=epochs)
            
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                
                src = batch.grapheme
                trg = batch.phoneme
                #trg = [trg len, batch size]

                self.optimizer.zero_grad()
                output = self.model(src, trg)
                if self.opt.model_type == 'LSTM':
                    output_dim = output.shape[-1]
                    #output = [trg len, batch size, output dim]
                    # TODO output.contiguous() 찍어보기
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].view(-1)

                elif self.opt.model_type == 'Transformer':
                    output, _ = output
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:,1:].contiguous().view(-1)

                # output = [(trg len - 1) * batch size, output dim]
                # trg = [(trg len - 1) * batch size]
                loss = self.criterion(output, trg)    
            
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                epoch_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.lr_scheduler.step()  # Update learning rate schedule
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps==0:
                        valid_loss = self.evaluate(valid_dataloader)
                        epochs.write(f"  global steps = {global_step}")
                        epochs.write(f"  valid loss = {valid_loss}")
                        epochs.write(f'  learning rate = {self.lr_scheduler.get_last_lr()}') # TODO

                        #TODO
                        training_stats.append(
                            {'epoch': epoch + 1,
                            'training_loss': epoch_loss / (step+1),
                            'valid_loss': valid_loss,
                            'steps': global_step
                            }
                        )
                    
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            self.save_model(best_valid_loss, epoch)
                            early_cnt = 0
                        else:
                            early_cnt += 1
                
                if 0 < self.args.max_steps < global_step or early_cnt > self.args.early_cnt:
                    break
            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            epochs.write(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | current lr : {self.lr_scheduler.get_last_lr()}')
            epochs.write(f"  total_steps = {global_step}")
            epochs.write(f"  train_loss = {epoch_loss / (step+1):.4f} | 'Train_PPL' = {math.exp(epoch_loss / (step+1)):.3f} ")
            epochs.write(f"  valid_loss = {valid_loss:.4f} | 'Val_PPL' = {math.exp(valid_loss):.3f}")

            if 0 < self.args.max_steps < global_step or early_cnt > self.args.early_cnt:
                df_stats = pd.DataFrame(data=training_stats, )
                df_stats = df_stats.set_index('epoch')
                df_stats.to_csv(f'{self.args.save_model_dir}/stats.csv', sep='\t', index=True)
                break
            
    def evaluate(self, valid_dataloader):
            
        self.model.eval()
        
        epoch_loss = 0
        with torch.no_grad():
        
            for step, batch in enumerate(progress_bar(valid_dataloader)):
                src = batch.grapheme
                trg = batch.phoneme

                output = self.model(src, trg)
                if self.opt.model_type == 'LSTM':
                    output_dim = output.shape[-1]
                    #output = [trg len, batch size, output dim]
                    # TODO output.contiguous() 찍어보기
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].view(-1)

                elif self.opt.model_type == 'Transformer':
                    output, _ = output
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:,1:].contiguous().view(-1)
                loss = self.criterion(output, trg)    

                epoch_loss += loss.item()
            
        return epoch_loss / (step+1)
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def save_model(self, loss, epoch):
        fpath = self.args.save_model_dir
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        torch.save({
            'epoch': epoch,
            'model_stat_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
            }, os.path.join(fpath, 'model.pt'))
        torch.save(self.args, os.path.join(fpath, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", fpath)
