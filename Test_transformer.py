'''

추론결과 성능 분석 및 한글자소로 변환
문장단위로 있는 것도 토큰을 분리하여 단어단위로 Error를 계산

@author: Jeongpil Lee (koreanfeel@gmail.com)
@created at : 2018. 08. 01.

'''

import json
import os
import torch
from tqdm import tqdm
import re
from functools import partial

def evaluate(input_file, output_file='evaluation_result_conv.json'):

    input_dir = os.path.dirname(input_file)

    input_dict = {}
    total_count = 0
    word_total_count = 0
    word_error_count = 0
    correct_count = 0
    output_list = []
    phone_total_count = 0
    phone_error_count = 0
    length_error_count = 0
    split_token = ' _ '
    Gr = 'grapheme'
    Ph = 'phoneme'
    Pr = 'predicted'
    Gr_p = 'original_pause'
    Pr_p = 'predicted_pause'



    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)

        total_count = len(data)

        for elem in data:
            g_sentence = elem[Gr] 
            p_sentence = elem[Ph] 
            hyp_sentence = elem[Pr] 

            # pause symbol 무시하고 성능계산
            p_sentence_tmp = ' '.join([p for p in p_sentence.split() if p != '/' and p != ','])
            hyp_sentence_tmp = ' '.join([h for h in hyp_sentence.split() if h != '/' and h != ','])
            g_words = g_sentence.replace(' ', '').split('_')
            p_words = p_sentence_tmp.split(' _ ')
            hyp_words = hyp_sentence_tmp.split(' _ ')
            p_words_with_pause = p_sentence.split(' _ ')
            hyp_words_with_pause = hyp_sentence.split(' _ ') 

            org_pause_sentence = []
            pred_pause_sentence = []

            if len(p_words) != len(hyp_words):
                length_error_count += 1

            for i, (g, p) in enumerate(zip(g_words, p_words)):

                g_word = g
                p_word = p
                p_phones = p.split()

                try: # 간혹 추론문장이 입력문장보다 짧은 경우가 있어서 예외처리 추가
                    hyp = hyp_words[i]
                    hyp_word = hyp

                except IndexError:
                    hyp_word = ''

                hyp_phones = hyp_word.split()


                # add pause
                org_pause_sentence.append(g_word)
                pred_pause_sentence.append(g_word)
                if len(g_word) > 0 and g_word[-1] in ['/', ',']:
                    org_pause_sentence.append(p_words_with_pause[i][-1])
                if len(p_word) > 0 and p_word[-1] in ['/', ',']:
                    pred_pause_sentence.append(hyp_words_with_pause[i][-1])

                word_total_count += 1
                phone_total_count += len(p_phones)

                is_correct = False
                if p_word == hyp_word:
                    is_correct = True
                else:
                    if len(p_phones) != len(hyp_phones):
                        phone_error_count += len(hyp_phones)
                    else:
                        for p_char, hyp_char in zip(p_phones, hyp_phones):
                            if p_char != hyp_char:
                                phone_error_count += 1

                if g_word not in input_dict:
                    if is_correct:
                        input_dict[g_word] = {'correct': 1, 'incorrect': 0, 'correct_results': [{p_word: hyp_word}], 'incorrect_results': []}
                    else:
                        input_dict[g_word] = {'correct': 0, 'incorrect': 1, 'correct_results': [], 'incorrect_results': [{p_word: hyp_word}]}
                else:
                    if is_correct:
                        input_dict[g_word]['correct'] += 1
                        if {p_word: hyp_word} not in input_dict[g_word]['correct_results']:
                            input_dict[g_word]['correct_results'].append({p_word: hyp_word})

                    else:
                        input_dict[g_word]['incorrect'] += 1
                        if {p_word: hyp_word} not in input_dict[g_word]['incorrect_results']:
                            input_dict[g_word]['incorrect_results'].append({p_word: hyp_word})

            if elem[Ph] != elem[Pr]:
                out_elem = {}
                out_elem[Gr] = ' '.join(g_words)
                out_elem[Ph] = p_sentence
                out_elem[Pr] = hyp_sentence
                out_elem[Gr_p] = ' '.join(org_pause_sentence)
                out_elem[Pr_p] = ' '.join(pred_pause_sentence)

                output_list.append(out_elem)


        true_count = 0
        errors = 0
        for key, elem in input_dict.items():
            correct_count += elem['correct']
            word_error_count += elem['incorrect']

            if elem['correct'] > 0 and elem['incorrect'] == 0:
                true_count += 1
            else:
                errors += 1


        print('\n### 단어단위 성능 - 중복미허용 ###')
        print("Words: %d" % len(input_dict))
        print("Errors: %d" % errors)
        print("WER: %.3f" % (float(errors) / len(input_dict)))
        print("Accuracy: %.3f" % float(1 - (errors / len(input_dict))))

        print('\n### 단어단위 성능 - 중복허용 ###')
        print('Words:', word_total_count)
        print('Errors:', word_error_count)
        print('WER: %.3f' % (word_error_count / word_total_count))
        print('Accuracy: %.3f' % (1 - (word_error_count / word_total_count)))

        print('\n### 음소단위 성능 ###')
        print('Phones:', phone_total_count)
        print('Errors:', phone_error_count)
        print('PER: %.3f' % (phone_error_count / phone_total_count))
        print('Accuracy: %.3f' % (1 - (phone_error_count / phone_total_count)))

        print('\n### 기타 ###')
        print('Total sentences :', total_count)
        print('Length errors :', length_error_count)

    output_path = os.path.join(input_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as fw:
        data_w = json.dumps(output_list, indent=4, sort_keys=True)
        data_w = bytes(data_w, 'utf-8').decode('unicode_escape')
        fw.write(data_w)
        _print = partial(print, file=fw, flush=True)

        
        _print('\n### 단어단위 성능 - 중복미허용 ###')
        _print("Words: %d" % len(input_dict))
        _print("Errors: %d" % errors)
        _print("WER: %.3f" % (float(errors) / len(input_dict)))
        _print("Accuracy: %.3f" % float(1 - (errors / len(input_dict))))

        _print('\n### 단어단위 성능 - 중복허용 ###')
        _print('Words:', word_total_count)
        _print('Errors:', word_error_count)
        _print('WER: %.3f' % (word_error_count / word_total_count))
        _print('Accuracy: %.3f' % (1 - (word_error_count / word_total_count)))

        _print('\n### 음소단위 성능 ###')
        _print('Phones:', phone_total_count)
        _print('Errors:', phone_error_count)
        _print('PER: %.3f' % (phone_error_count / phone_total_count))
        _print('Accuracy: %.3f' % (1 - (phone_error_count / phone_total_count)))

        _print('\n### 기타 ###')
        _print('Total sentences :', total_count)
        _print('Length errors :', length_error_count)

    print('\nFile saved to :', output_path)


def translate(batch, dataset, model, device):
    '''
    transformer interaction show
    '''
    # model.eval()
    src_mask = model.make_src_mask(batch.grapheme)
    
    # with torch.no_grad():
    enc_src = model.encoder(batch.grapheme, src_mask)
    
    pho_idx = [dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.init_token]]
    for i in range(250):
        pho_tensor = torch.LongTensor(pho_idx).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(pho_tensor)
        
        # with torch.no_grad():
        output, _ = model.decoder(pho_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()
        pho_idx.append(pred_token)

        if pred_token == dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.eos_token]:
            break
    pho_tokens = [dataset.P_FIELD.vocab.itos[i] for i in pho_idx]
    
    return pho_tokens[1:-1]


def translate_sentence(sentence, dataset, model, device):

    # model.eval()

    tokens = list(re.sub(' ', '_', sentence))

    tokens = [dataset.G_FIELD.init_token] + tokens + [dataset.G_FIELD.eos_token]
    print(tokens)

    src_indexes = [dataset.G_FIELD.vocab.stoi[token] for token in tokens]
    print(dataset.G_FIELD.vocab.stoi)
    
    print(src_indexes)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    print(src_tensor)
    
    src_mask = model.make_src_mask(src_tensor)
    print(src_mask)
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.init_token]]

    for i in range(250):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        print(trg_mask)
        # with torch.no_grad():
        output, attention = model.decoder(
            trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == dataset.P_FIELD.vocab.stoi[dataset.P_FIELD.eos_token]:
            break
    print(dataset.P_FIELD.vocab.stoi)
    print(dataset.P_FIELD.vocab.itos)
    print(trg_indexes)
    
    trg_tokens = [dataset.P_FIELD.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:-1]


def test(dataset, model, device, model_path):
    # TODO 모델 인자
    model.load_state_dict(torch.load(f'{model_path}/model_best.pt'))
    _, _, test_iter = dataset.build_iterator()
    # while(1):
    #     sent = input('입력: ')
    #     output = translate_sentence(sent, dataset, model, device)
    #     print(' '.join(output))
    out = []
    # for i, batch in enumerate(test_iter):
    for batch in tqdm(test_iter):
        g_field = batch.dataset.fields['grapheme']
        p_field = batch.dataset.fields['phoneme']
        gra = batch.grapheme.squeeze(0).data.tolist()[1:-1]
        pho = batch.phoneme.squeeze(0).data.tolist()[1:-1]
        pred = translate(batch, dataset, model, device)
        data = {}
        data['grapheme'] = ' '.join([g_field.vocab.itos[g] for g in gra])
        data['phoneme'] = ' '.join([p_field.vocab.itos[p] for p in pho])
        data['predicted'] = ' '.join(pred)
        out.append(data)

    #     # print("> {}\n= {}\n< {}\n".format(' '.join([g_field.vocab.itos[g] for g in gra]),
    #     # ' '.join([p_field.vocab.itos[p] for p in pho]),
    #     # ' '.join(pred)))

    #     # if i == 3:
    #     #     break

    with open(f'{model_path}/test_out.json', 'w', encoding='utf-8') as wf:
        json.dump(out, wf, ensure_ascii=False, indent='\t')

    evaluate(f'{model_path}/test_out.json')



if __name__ == "__main__":
    input_file = 'checkpoints/2020-08-26T13:37:07_transformer/test_out.json'
    # input_file = 'model/eng/evaluation_result_iter_13500.json'
    evaluate(input_file)
