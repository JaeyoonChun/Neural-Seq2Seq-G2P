import torch
import random
import os
import logging
import pickle
import json

def get_angles(pos, i, d_model):
  angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / torch.tensor(d_model, dtype=torch.float32))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(torch.arange(position).unsqueeze(1),
                            torch.arange(d_model).unsqueeze(0),
                            d_model)

  # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
  angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

  # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
  angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads.unsqueeze(0)
  return pos_encoding

def sequence_mask(src): 
  #src = [batch size, src len]
  
  src_mask = (src != 1).unsqueeze(1).unsqueeze(2)

  #src_mask = [batch size, 1, 1, src len]

  return src_mask

def set_seeds():
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def load_device(args):
  if args.world_size == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device_num}'
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def normalize(src, trg):
    src = src.rstrip('\n')
    trg = trg.rstrip('\n')

    src = ''.join(list(src.replace(' ', '_')))
    trg = ' '.join(list(trg.replace(' ', '_')))

    return src, trg

def dataset_process(args):
  data_dir = args.data_dir
  with open(os.path.join(data_dir, f"{args.data_type}_trainset.pkl"), 'rb') as f,\
    open(os.path.join(data_dir, f"{args.data_type}_testset.pkl"), 'rb') as fw:
    trainset = pickle.load(f)
    testset = pickle.load(fw)
    
  total = len(trainset)
  ratio = (2, 1, 1)

  ratio = [total//sum(ratio) * r for r in ratio]
  print(ratio)
  train, valid, test = [], [], []
  for r, (topic, lines) in zip(ratio, trainset.items()):
    if r == 0:
      continue
    
    lines = lines[:r]
    valid += [normalize(src=l['new_src'], trg=l['new_trg']) for l in lines[:len(lines)//10]]
    train += [normalize(src=l['new_src'], trg=l['new_trg']) for l in lines[len(lines)//10:]]

  for topic, lines in testset.items():      
    test += [normalize(src=l['new_src'], trg=l['new_trg']) for l in lines]
      
  train = sorted(train, key=lambda x: len(x[0]))
  valid = sorted(valid, key=lambda x: len(x[0]))
  test = sorted(test, key=lambda x: len(x[0]))

  train_save = []
  for src, trg in train:
    train_save.append(
      {
          "G": src,
          "P": trg
      }
    )
  
  val_save = []
  for src, trg in valid:
    val_save.append(
      {
        "G": src,
        "P": trg
      }
    )

  test_save = []
  for src, trg in test:
    test_save.append(
      {
        "G": src,
        "P": trg
      }
    )

  with open(os.path.join(data_dir, f"{args.data_type}_train.json"), 'w', encoding='utf-8') as f,\
    open(os.path.join(data_dir, f"{args.data_type}_dev.json"), 'w', encoding='utf-8') as fw,\
    open(os.path.join(data_dir, f"{args.data_type}_test.json"), 'w', encoding='utf-8') as ft:
    json.dump(train_save, f, ensure_ascii=False, indent='\t')
    json.dump(val_save, fw, ensure_ascii=False, indent='\t')
    json.dump(test_save, ft, ensure_ascii=False, indent='\t')
