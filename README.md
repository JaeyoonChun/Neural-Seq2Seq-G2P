# Sogang English G2P (Grapheme-to-Phoneme)

## Requirements
* Python 3.6+
* torch 1.6

아래 명령을 이용하여 필요한 패키지를 설치하시면 됩니다.

```
pip install -r requirements.txt
```

## Running

**train**

```
python main.py --model_mode [LSTM or Transformer]
```
**Test**

```
python prediction.py --model_mode [LSTM or Transformer]
```