# Multi-modal Korean Emotion Recognition with consistency regularization
This is pytorch implementation of the paper, "Multi-modal Korean Emotion Recognition with consistency regularization" which is submitted to [Korean Institute of Industrial Engineers(KIIE)](http://kiie.org/).

## Dataset
We experiment our model with [`Multi-modal Video Dataset`](https://aihub.or.kr/aidata/137) released from [AIhub](AIhub)

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

### 3. preprocessing
- `./src/configs/iemocap_preproc.yaml`
    - use_bert: True: 텍스트모덜 벡터로 BERT last encoder layer 사용, 아니면 glove vectors사용
    - use_wav2vec2: 오디오모덜 벡터로 Wav2vec2.0 last encoder layer 사용, 아니면 MFCC vectors사용

```shell
python src/preprocess/preproc.py
```
- in:
    - `iemocap_data.pkl` (raw_data)
- out
    - `iemocap_data_feature_bert_True_wav2_True.pkl`


# multimodal Classifier
- `./src/configs/model.yaml`
    - `only_audio (True)`: 오디오 modal만 활용한 Transformer 학습
    - `only_text (True)`: 텍스트 modal만 활용한 Transformer 학습
    - `only_audio (False) & only_text (False)`: 두개의 modal를 활용한 Transformer 학습

```shell
python main.py
```
- in:
    - `iemocap_data_feature_bert_True_wav2_True.pkl`
- out
    - `./models/bert_wav2_modal/test_results.csv`

