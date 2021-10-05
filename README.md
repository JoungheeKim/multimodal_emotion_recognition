# Multi-modal Korean Emotion Recognition with consistency regularization
This is pytorch implementation of the paper, "Multi-modal Korean Emotion Recognition with consistency regularization" which is submitted to [Korean Institute of Industrial Engineers(KIIE)](http://kiie.org/).

## Dataset
We experiment our model with [`Multi-modal Video Dataset`](https://aihub.or.kr/aidata/137) released from [AIhub](AIhub)

## Pre-trained Model
We use a lot of pre-trained model to leverage performance with limited resources.
You can download each pre-trained model to visit their offical repository.
Please download pre-trained model and put them in some directory to use as a part of our system.

1. [Wav2vec 2.0](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) : An end-to-end framework for ASR released by Fairseq.
2. [VQ-wav2vec](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) : A framework where quantization module is applied to capture patterns in speech.
3. [HanBERT](https://github.com/monologg/HanBert-Transformers) : Korean BERT Model pre-trained with Korean WIKI and Book corpus.
4. [FastText](https://github.com/ratsgo/embedding) : Word Embedding which is trained with Korean WIKI dataset.


## Install
### 1. Docker
We support `Dockerfile` to build virtual environment to run our system.

### 2. Fairseq Install
Our system use Wav2vec 2.0 released by `fairseq`.
Therefore, it is necessary to install fairseq system before install our system.
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### 3. Basic Requirement
```bash
## OS Library for audio
apt-get install libsndfile1-dev

## Python Library
pip install librosa
pip install pydub
pip install konlpy
pip install soynlp
pip install conformer
pip install transformers

## install torchaudio following the insturction(https://github.com/pytorch/audio)
## make sure it is right version to your pytorch library
pip install torchaudio==0.7.2
```

### 4. Install our system
Install 
```bash
git clone https://github.com/JoungheeKim/multimodal_emotion_recognition.git
cd multimodal_emotion_recognition
python setup.py develop
```

## Preprocessing
### 1. Convert Video to Audio & make information Dataframe
Multi-modal Video Dataset is composed of `Video` and `Json`. To preprocess these files into organized form, run script file below:
```bash
bash script/video_to_audio/video_to_audio.sh
```
### 2. Generate Speech to Text(STT) Trascription
To use STT API from other system, you need to visit official site first.
1. [KAKAO STT API](https://speech-api.kakao.com/)
2. [GOOGLE STT API](https://cloud.google.com/speech-to-text?hl=ko)

We support some of sample code to show how to use these module in our system.
However, there is lot of things ,such as private key and support licence, to use their system.
Therefore, we recomment you to look around there official site first, and then watch our code.

We give some samples to run STT API
```bash
## Run kakao
## Before run the bash file, you need to make sure that `kakao_key` is implied in kakao_api.py file
bash script/speech_to_text/run_kakao.sh
```

### 3. Split data and Label Preprocessing
```bash
bash script/preprocess/preprocess.sh
```

## Train Multi modal
We support diverse multi-modal model.
You can find details in our paper.
```bash
## run Shallow fusion model with wav2vec 2.0 and hanbert


## run Multi-modal model with wav2vec 2.0 and hanbert


## run Deep fusion model with wav2vec 2.0 and hanbert


```
