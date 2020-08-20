import json
import os
import re
import sys

g_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", 
          "t", "u", "v", "w", "x", "y", "z", "'", "<unk>"]

p_list = [",", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", 
          "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", 
          "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", 
          "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW0", "UW1", 
          "UW2", "V", "W", "Y", "Z", "ZH", "spn"] #, "/"]


def conv_english_to_code(text, g_or_p='p', split_token='_'):
    split_token = split_token.strip()
    words = text.strip().split()
    out_seq = ''

    if g_or_p == 'g':
        for k, word in enumerate(words):
            if word == '<unk>':
                out_seq += f'{word} {split_token} '
                continue

            for j, char in enumerate(word):
                out_seq += char

                if j < len(word) - 1:
                    out_seq += ' '

            if k < len(words) - 1:
                out_seq = out_seq + ' ' + split_token + ' '

        return out_seq
    else:
        return text

def get_file(fpath, type):        
    with open(os.path.join(fpath, f'librispeech_{type}-clean.json'), "r", encoding='utf-8') as f:
        data = json.load(f)

    sent_out = open(os.path.join(fpath, f"sent_{type}.txt"), "w", encoding='utf-8')

    bucket_limit = 10000  # 문장 길이 제한
    count = 0
    split_token = ' _ '
    max_sequence_len = 0 # grapheme sequence의 최대 길이

    print("########  Start - Sentence  ########")
    s = []
    for elem in data:
        file = None
        if 'file' in elem:
            file = elem['file']

        g_text = elem['G']
        result = re.findall(r'[^a-z\' ]', g_text)

        p_text = elem['P']
        result = re.findall(r'[^A-Z0-9_\/, ]', p_text)

        try:
            temp_G = conv_english_to_code(g_text, g_or_p='g', split_token=split_token)

            if len(temp_G) >= bucket_limit:
                count += 1
                print('exceed!')
                continue
            temp_P = conv_english_to_code(p_text, g_or_p='p', split_token=split_token)
            
            # 문장별로 max_sequence_len을 구함
            sentence_len = len(temp_P.split())
            if sentence_len > max_sequence_len:
                max_sequence_len = sentence_len
        except:
            print('\nError occured!')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            import traceback
            traceback.print_exc()

            if file:
                print(file)
            print(g_text)
            print(p_text)
            exit()
            continue
        # TODO sequence 길이대로 sort하여 
        s.append(temp_G + "##" + temp_P + "\n")
        # sent_out.write(temp_G + "##" + temp_P + "\n")

        count += 1
    s = sorted(s, key=lambda x: len(x))
    sent_out.writelines(s)
    sent_out.close()
    print(f"Done : sent_{type}.txt")
    print("count :", count)

    print('max_sequence_len :', max_sequence_len)
    print("########  Complete - Sentence  ########")