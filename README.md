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
python main.py --do_train --version [save_model_path]
```

**test**

```
python main.py --do_test --version [save_model_path]
```



## Copyright

본 프로젝트는 TensorFlow나 Numpy 등 공개소프트웨어를 이용하였으나, 데이터 처리 방법 및 모델 구현에 대한 저작권은 서강대학교 지능형 음성대화 인터페이스 연구실에 있습니다.

