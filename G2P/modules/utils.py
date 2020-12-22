import torch
import random
import os
import logging

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
