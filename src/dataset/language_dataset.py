from dataclasses import _MISSING_TYPE, dataclass, field
import io
import os
from glob import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import yaml
import torchaudio
import pandas as pd
import torch
from .base_dataset import BaseProcessor
from transformers import BertModel
from src.tokenization_hanbert import HanBertTokenizer
from .base_dataset import FeatureConfig
from .word_process import Word2Vec
from dataclasses import _MISSING_TYPE, dataclass, field
import numpy as np
import logging

@dataclass
class LanguageFeatureConfig(FeatureConfig):
    name:str='raw'
    feature_dim:int = 1
    column_name:str = 'stt'
    add_column_name: str = ''
    train_column_name: str = ''

@dataclass
class HanTokenFeatureConfig(LanguageFeatureConfig):
    name: str = 'hantoken'
    bert_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/HanBert-54kN-torch'
    feature_dim:int=1

@dataclass
class HanBertFeatureConfig(LanguageFeatureConfig):
    name: str = 'hanbert'
    bert_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/HanBert-54kN-torch'
    feature_dim:int=768

@dataclass
class GloveFeatureConfig(LanguageFeatureConfig):
    name: str = 'glove'
    vocab_path:str='/code/gitRepo/sentiment_speech/pretrained_model/word-embeddings/glove/glove.txt'
    tokenizer_name:str = 'mecab'
    method: str = 'glove'
    feature_dim:int=100

@dataclass
class FasttextFeatureConfig(LanguageFeatureConfig):
    name: str = 'fasttext'
    vocab_path: str = '/code/gitRepo/sentiment_speech/pretrained_model/word-embeddings/fasttext-jamo/fasttext-jamo.vec'
    model_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/word-embeddings/fasttext-jamo/fasttext-jamo.bin'
    tokenizer_name:str = 'mecab'
    method:str='fasttext-jamo'
    feature_dim:int=100

@dataclass
class Word2vecFeatureConfig(LanguageFeatureConfig):
    name: str = 'word2vec'
    vocab_path: str = '/code/gitRepo/sentiment_speech/pretrained_model/word-embeddings/word2vec/word2vec'
    tokenizer_name: str = 'mecab'
    method: str = 'word2vec'
    feature_dim:int=100


class LanguageProcessor(BaseProcessor):
    def __init__(self, config:LanguageFeatureConfig):
        super(LanguageProcessor, self).__init__(config)
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.feature_name = config.name
        self.column_name = config.column_name
        self.add_column_name = config.add_column_name
        self.train_column_name = config.train_column_name

    def _load_model(self):
        config = self.config
        if self.feature_name == 'raw':
            pass

        elif self.feature_name == 'hanbert':
            assert config.bert_path != '' and config.bert_path is not None, 'BERT 모델이 저장된 폴더 또는 이름을 정확히 다시한번 확인해 보세요. [{}]'.format(
                config.bert_path)
            self.model = BertModel.from_pretrained(config.bert_path)
            self.model.to(self.device)
            self.tokenizer = HanBertTokenizer.from_pretrained(config.bert_path)

        elif self.feature_name == 'hantoken':
            assert config.bert_path != '' and config.bert_path is not None, 'BERT 모델이 저장된 폴더 또는 이름을 정확히 다시한번 확인해 보세요. [{}]'.format(
                config.bert_path)
            self.tokenizer = HanBertTokenizer.from_pretrained(config.bert_path)

        else:
            raise ValueError('Need to add audio feature [{}]'.format(config.name))


    def _release_model(self):
        if hasattr(self, 'audio2vec'):
            self.model.to('cpu')
            del self.model

    @torch.no_grad()
    def convert(self, data):
        if self.feature_name == 'hanbert':
            input_args = self.tokenizer(data, return_tensors="pt")
            input_args = input_args.to(self.device)
            outputs = self.model(**input_args)
            # get last layer
            outputs = outputs[0]
            return outputs.squeeze(0).cpu().numpy()

        if self.feature_name == 'hantoken':
            tokens = self.tokenizer.encode(data)
            return np.array(tokens)

        return data


    def convert_data(self, script_df, split_name=None, **kwargs):
        ## load model
        self._load_model()

        ## for tqdm
        tqdm.pandas()

        column_name = self.column_name
        if split_name == 'train' and self.train_column_name !='':
            column_name = self.train_column_name

        logging.info('convert text to language feature by {}'.format(self.feature_name))
        script_df[column_name] = script_df.progress_apply(
            lambda x: self.convert(x[column_name]), axis=1)

        if self.add_column_name != '' and self.add_column_name in script_df.columns:
            script_df[self.add_column_name] = script_df.progress_apply(
                lambda x: self.convert(x[self.add_column_name]), axis=1)

        ## Release model
        self._release_model()
        return script_df

    def get_data(self, script_df, split_name=None):
        column_name = self.column_name
        if split_name=='train' and self.train_column_name !='':
            column_name = self.train_column_name

        return script_df[column_name].values.tolist()

    def get_add_data(self, script_df):
        if self.add_column_name != '' and self.add_column_name in script_df.columns:
            return script_df[self.add_column_name].values.tolist()
        return None
