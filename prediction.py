import json
import os
import torch
from tqdm import tqdm
import re
from functools import partial
from pprint import pprint
import logging
from attrdict import AttrDict

from G2P.modules.utils import set_seeds, load_device, init_logger
from G2P.build_model import build_model
from G2P.data_loader import build_dataset
import torch.nn.functional as F
import argparse

logger = logging.getLogger(__name__)

def translate_Transformer(batch, fields, model, device, max_decode_len):
    '''
    transformer sentence translate
    '''
    src = batch.grapheme
    TRG_FIELD = fields['phoneme']

    enc_src = model.encoder(src)
    trg_idx = [TRG_FIELD.vocab.stoi[TRG_FIELD.init_token]]
    
    for i in range(max_decode_len):
        trg_tensor = torch.LongTensor(trg_idx).unsqueeze(0).to(device)
        
        output, _ = model.decoder(trg_tensor, enc_src, src=src, predict=True)

        pred_token = output.argmax(2)[:,-1].item()
        trg_idx.append(pred_token)

        if pred_token == TRG_FIELD.vocab.stoi[TRG_FIELD.eos_token]:
            break

    trg_tokens = [TRG_FIELD.vocab.itos[i] for i in trg_idx]
    
    return trg_tokens[1:-1]

def translate_Transformer_beam_search(batch, fields, model, device, beam_size, max_decode_len):
    '''
    Transformer beam search
    '''
    src = batch.grapheme
    TRG_FIELD = fields['phoneme']

    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src)   
    # enc_src = [1, src len, hid dim]

    init_token = TRG_FIELD.init_token
    eos_token = TRG_FIELD.eos_token
    
    hypotheses = [[init_token]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    time_step = 0
    while len(completed_hypotheses) < beam_size and time_step < max_decode_len:
        time_step += 1
        hyp_num = len(hypotheses)

        enc_src_expand = enc_src.expand(hyp_num, -1, -1)
        src_expand = src.expand(hyp_num, -1)
        
        trg_tensor = torch.tensor([[TRG_FIELD.vocab.stoi[h] for h in hyp] for hyp in hypotheses], dtype=torch.long, device=device)
     
        output, _ = model.decoder(trg_tensor, enc_src_expand, src=src_expand, predict=True) 
        # attention = [hyp_num, n heads, trg len, src len]       
        # output = [hyp_num, trg len, output dim]
        
        trg_vovab_size = output.shape[-1]

        new_scores = F.log_softmax(output[:, -1, :], dim=-1)
        # output[:, -1, :] 인 이유
        # Transformer는 타임스탭 마다 한 토큰씩 입력하지 않고 문장 전체를 입력하므로 output 도 trg len 만큼 나오게 된다.
        # beam search를 위해서는 현재 타임스텝에서 예측한 토큰의 score를 알아야 한다.
        # 따라서 output = [hyp_num, trg len, output dim] 일 때, trg len에서 마지막 토큰만 빼내어 softmax에 태운다

        prev_scores = hyp_scores.unsqueeze(1).expand_as(new_scores)
        new_scores = prev_scores + new_scores
        flatten_scores = new_scores.view(-1) # [vocab size * hyp_num]
        
        live_hyp_num = beam_size - len(completed_hypotheses)
        best_scores, best_scores_id = torch.topk(flatten_scores, k=live_hyp_num)

        which_prev_hyps = best_scores_id // trg_vovab_size 
        # 나눠주는 이유: 
        # 이전 hyp에서 가지를 뻗어서 다음 단어를 예측 하는데, 
        # 예측된 단어들 중 top k가 이전 hyp 중 어디서 뻗어서 나온 건지 알아야 된다.
        # 그런데 현재 id가 flatten( [vocab size * hyp_num] ) 이므로 
        # 어떤 단어에서 나왔는지 파악하기 위해서 vocab size로 나누어 주는 것이다.
        # example)
        # new_scores = [[0.1, 0.1, 0.8], 0번째 단어에서 뻗어나온 다음 단어
        #               [0.1, 0.8, 0.1], 1번째 단어에서 뻗어나온 다음 단어
        #               [0.2, 0.4, 0.4]] 2번째 단어에서 뻗어나온 다음 단어
        # vocab size = 3, hyp_num = 3. k=2
        # flatten_scores = [0.1, 0.1, 0.8, 0.1, 0.8, 0.1, 0.2, 0.4, 0.4]
        #                  k=2 이므로 2개의 0.8 을 고를 것이다. 
        # best_scores = [0.8, 0.8]
        # best_scores_id = [2, 4]
        #                  best_scores_id 만으로는 몇번째 단어에서 뻗어나온건지 알 수 없으므로,
        # prev_hyp_ids = best_scores_id // trg_vovab_size
        #              = [2, 4] // 3 
        #              = [0, 1]
        # ===> 0, 1번째 단어에서 예측된 단어임을 알 수 있음
        next_hyp_ids = best_scores_id % trg_vovab_size
        # best_scores_id를 그대로 가져오기 위해.

        next_hypotheses = []
        next_hyp_scores = []

        for which_prev_hyp, next_hyp_id, best_score in zip(which_prev_hyps, next_hyp_ids, best_scores):
            which_prev_hyp = which_prev_hyp.item()
            next_hyp_id = next_hyp_id.item()
            best_score = best_score.item()

            next_hyp_word = TRG_FIELD.vocab.itos[next_hyp_id]
            next_hyp_sent = hypotheses[which_prev_hyp] + [next_hyp_word]

            if next_hyp_word == eos_token:
                completed_hypotheses.append((next_hyp_sent[1:-1], best_score))
                continue

            next_hypotheses.append(next_hyp_sent)
            next_hyp_scores.append(best_score)
        
        if len(completed_hypotheses) == beam_size:
            break

        hypotheses = next_hypotheses
        hyp_scores = torch.tensor(next_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append((hypotheses[0][1:], hyp_scores[0].item()))

    completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x[1], reverse=True)
    return completed_hypotheses

def translate_LSTM(batch, fields, model, device, max_decode_len):
    '''
    biLSTM sentence translate
    '''
    src = batch.grapheme
    TRG_FIELD = fields['phoneme']
    init_token = TRG_FIELD.init_token
    eos_token = TRG_FIELD.eos_token

    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(src)
    enc_src, h, c = encoder_outputs
   
    trg_idx = [TRG_FIELD.vocab.stoi[init_token]]
    for i in range(max_decode_len):
        teg_tensor = torch.LongTensor(trg_idx).to(device)
       
        with torch.no_grad():
            o, h, c = model.decoder._run(teg_tensor[-1:], h, c, enc_src)
        pred_token = o.argmax(1)
        trg_idx.append(pred_token)
        if pred_token == TRG_FIELD.vocab.stoi[eos_token]:
            break

    trg_tokens = [TRG_FIELD.vocab.itos[i] for i in trg_idx]
    
    return trg_tokens[1:-1]

def translate_LSTM_beam_search(batch, fields, model, device, beam_size, max_decode_len):
    '''
    biLSTM beam search
    '''
    src = batch.grapheme
    TRG_FIELD = fields['phoneme']

    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(src)
    enc_src, h_in, c_in = encoder_outputs
    # outputs = [src len, 1, hid dim * 2(bidirectional)]
    # hidden = [1, dec hid dim]
    # cell = [1, dec hid dim]
    
    init_token = TRG_FIELD.init_token
    eos_token = TRG_FIELD.eos_token
    
    hypotheses = [[init_token]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
    completed_hypotheses = []

    time_step = 0
    while len(completed_hypotheses) < beam_size and time_step < max_decode_len:
        time_step += 1
        hyp_num = len(hypotheses)

        exp_src_encodings = enc_src.expand(-1, hyp_num, -1)
        trg_tensor = torch.tensor([TRG_FIELD.vocab.stoi[h[-1]] for h in hypotheses], dtype=torch.long, device=device)
        #TODO trg_tensor shape 확인

        o, h, c = model.decoder._run(trg_tensor, h_in, c_in, exp_src_encodings) 
        # o = [hyp_num, vocab size]
        # h = [hyp_num, hid dim]
        # c = [hyp_num, hid dim]
        
        trg_vovab_size = o.shape[-1]

        new_scores = F.log_softmax(o, dim=-1) # 각 예측된 단어에 softmax 태우기
        prev_scores = hyp_scores.unsqueeze(1).expand_as(new_scores)
        new_scores = prev_scores + new_scores
        flatten_scores = new_scores.view(-1) # [vocab size * hyp_num]
        
        live_hyp_num = beam_size - len(completed_hypotheses)
        best_scores, best_scores_id = torch.topk(flatten_scores, k=live_hyp_num)

        which_prev_hyps = best_scores_id // trg_vovab_size 
        # 나눠주는 이유: 
        # 이전 hyp에서 가지를 뻗어서 다음 단어를 예측 하는데, 
        # 예측된 단어들 중 top k가 이전 hyp 중 어디서 뻗어서 나온 건지 알아야 된다.
        # 그런데 현재 id가 flatten( [vocab size * hyp_num] ) 이므로 
        # 어떤 단어에서 나왔는지 파악하기 위해서 vocab size로 나누어 주는 것이다.
        # example)
        # new_scores = [[0.1, 0.1, 0.8], 0번째 단어에서 뻗어나온 다음 단어
        #               [0.1, 0.8, 0.1], 1번째 단어에서 뻗어나온 다음 단어
        #               [0.2, 0.4, 0.4]] 2번째 단어에서 뻗어나온 다음 단어
        # vocab size = 3, hyp_num = 3. k=2
        # flatten_scores = [0.1, 0.1, 0.8, 0.1, 0.8, 0.1, 0.2, 0.4, 0.4]
        #                  k=2 이므로 2개의 0.8 을 고를 것이다. 
        # best_scores = [0.8, 0.8]
        # best_scores_id = [2, 4]
        #                  best_scores_id 만으로는 몇번째 단어에서 뻗어나온건지 알 수 없으므로,
        # prev_hyp_ids = best_scores_id // trg_vovab_size
        #              = [2, 4] // 3 
        #              = [0, 1]
        # ===> 0, 1번째 단어에서 예측된 단어임을 알 수 있음
        next_hyp_ids = best_scores_id % trg_vovab_size
        # best_scores_id를 그대로 가져오기 위해.

        next_hypotheses = []
        which_hyps_alive = []
        next_hyp_scores = []

        for which_prev_hyp, next_hyp_id, best_score in zip(which_prev_hyps, next_hyp_ids, best_scores):
            which_prev_hyp = which_prev_hyp.item()
            next_hyp_id = next_hyp_id.item()
            best_score = best_score.item()

            next_hyp_word = TRG_FIELD.vocab.itos[next_hyp_id]
            next_hyp_sent = hypotheses[which_prev_hyp] + [next_hyp_word]

            if next_hyp_word == eos_token:
                completed_hypotheses.append((next_hyp_sent[1:-1], best_score))
                continue

            next_hypotheses.append(next_hyp_sent)
            which_hyps_alive.append(which_prev_hyp) # 다음 단어가 예측되었으므로 살아있는 단어
            next_hyp_scores.append(best_score)
        
        if len(completed_hypotheses) == beam_size:
            break

        which_hyps_alive = torch.tensor(which_hyps_alive, dtype=torch.long, device=device)
        h_in = h[which_hyps_alive]
        c_in = c[which_hyps_alive]

        hypotheses = next_hypotheses
        hyp_scores = torch.tensor(next_hyp_scores, dtype=torch.float, device=device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append((hypotheses[0][1:], hyp_scores[0].item()))

    completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x[1], reverse=True)
    return completed_hypotheses


def test(model, fields, device, test_iter, args, opt):

    checkpoint = torch.load(f'{args.save_model_dir}/model.pt')
    model.load_state_dict(checkpoint['model_stat_dict'])

    out = []
    i = 0
    for batch in tqdm(test_iter):
        if opt.model_type == 'Transformer':
            src = batch.grapheme.squeeze(0).data.tolist()
            trg = batch.phoneme.squeeze(0).data.tolist()[1:-1]
            if args.beam_search:
                pred = translate_Transformer_beam_search(
                    batch, fields, model, device, args.beam_size, args.max_decode_len)
                pred = pred[0][0] # top 1 prediction
            else:
                # Greedy search
                pred = translate_Transformer(batch, fields, model, device, args.max_decode_len)
            
        if opt.model_type == 'LSTM':
            src = batch.grapheme.squeeze(1).data.tolist()
            trg = batch.phoneme.squeeze(1).data.tolist()[1:-1]
            if args.beam_search:
                pred = translate_LSTM_beam_search(
                    batch, fields, model, device, args.beam_search, args.max_decode_len)
                pred = pred[0][0] # top prediction
            else:
                # Greedy search
                pred = translate_LSTM(batch, fields, model, device, args.max_decode_len)

        data = {}
        data['grapheme'] = ' '.join([fields['grapheme'].vocab.itos[s] for s in src[1:-1]])
        data['phoneme'] = ' '.join([fields['phoneme'].vocab.itos[t] for t in trg])
        data['predicted'] = ' '.join(pred)
        out.append(data)

    test_file = f'test_out_beam_{args.beam_size}.json' if args.beam_search else 'test_out_greedy.json'
    with open(os.path.join(args.save_model_dir, test_file), 'w', encoding='utf-8') as wf:
        json.dump(out, wf, ensure_ascii=False, indent='\t')
    
def main(opt):
    model_args_path = os.path.join('G2P/config', opt.model_type+'.json')
    test_args_path = os.path.join('G2P/config', 'test_config.json')
    with open(model_args_path, 'r', encoding='utf-8') as f:
        model_args = AttrDict(json.load(f))
    with open(test_args_path, 'r', encoding='utf-8') as f:
        test_args = AttrDict(json.load(f))

    init_logger()
    set_seeds()
    device = load_device(test_args)

    (_, _, test_iter), fields = build_dataset(opt, test_args, device, model_args.vectors)
    model = build_model(model_args, opt, fields, device)
    model = model.to(device)
    test(model, fields, device, test_iter, test_args, opt)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='LSTM', required=True, type=str)
    opt = parser.parse_args()
    main(opt)