# coding=utf-8
# Copyright 2020 The JoungheeKim All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from pathlib import Path
import pandas as pd
from hydra.core.config_store import ConfigStore
from src.process.label_preprocess import FiveLabelPreprocessConfig, SevenLabelPreprocessConfig, Multimodal8PreprocessConfig, Multimodal7PreprocessConfig, Multimodal6PreprocessConfig, Multimodal4PreprocessConfig, Multimodal5PreprocessConfig, Synthesis7PreprocessConfig
from src.process.split_preprocess import SplitPreprocessConfig
from src.dataset.audio_dataset import MfccFeatureConfig, AudioFeatureConfig, Wav2vecFeatureConfig, VQTokenFeatureConfig, VQWav2vecFeatureConfig, MelFeatureConfig, ExceptFeatureConfig
from src.dataset.language_dataset import HanBertFeatureConfig, HanTokenFeatureConfig, LanguageFeatureConfig, FasttextFeatureConfig
from src.speech_to_text.kakao_api import KaKaoConfig
from src.speech_to_text.google_api import GoogleConfig
from src.speech_to_text.clova_api import ClovaConfig
from src.model.wav2keyword import Wav2KeywordConfig
from src.model.conformer import ConformerConfig
from src.model.lm_transformer import LMTransformerConfig
from src.model.audio_transformer import AudioTransformerConfig
from src.model.ad_lm_transformer import AudioLanguageTransformerConfig
from src.model.audio_lm_shallow import AudioLanguageShallowConfig
from src.model.audio_lm_shallow_point import AudioLanguageShallowPointConfig
from src.model.audio_lm_shallow_multi import AudioLanguageShallowMultiConfig
from src.model.audio_lm_shallow_multi_ver2 import AudioLanguageShallowMultiV2Config
from src.model.audio_lm_shallow_pointer import AudioLanguageShallowPointerConfig
from src.model.ad_lm_transformer_pointer import AudioLanguageTransformerPointerConfig
from src.model.audio_lm_deep import AudioLanguageDeepConfig
from src.model.audio_lm_deep_pointer import AudioLanguageDeepPointerConfig
from src.model.audio_lm_deep_pointer_ver2 import AudioLanguageDeepPointerV2Config

from src.configs import TrainConfig, PreprocessConfig, CallAPIConfig, EvalConfig

from src.configs import DefaultConfig

from datetime import datetime

def init():
    cs = ConfigStore.instance()

    ## base
    cs.store(group="base", name='train', node=TrainConfig)
    cs.store(group="base", name='eval', node=EvalConfig)
    cs.store(group="base", name='preprocess', node=PreprocessConfig)
    cs.store(group="base", name='api', node=CallAPIConfig)

    ## model
    cs.store(group="model", name="conformer", node=ConformerConfig)
    cs.store(group="model", name="wav2keyword", node=Wav2KeywordConfig)
    cs.store(group="model", name="lm_transformer", node=LMTransformerConfig)
    cs.store(group="model", name="audio_transformer", node=AudioTransformerConfig)
    cs.store(group="model", name="multimodal_transformer", node=AudioLanguageTransformerConfig)
    cs.store(group="model", name="multimodal_transformer_pointer", node=AudioLanguageTransformerPointerConfig)
    cs.store(group="model", name="shallow_fusion", node=AudioLanguageShallowConfig)
    cs.store(group="model", name="shallow_fusion_point", node=AudioLanguageShallowPointConfig)
    cs.store(group="model", name="shallow_fusion_pointer", node=AudioLanguageShallowPointerConfig)
    cs.store(group="model", name="shallow_fusion_multi", node=AudioLanguageShallowMultiConfig)
    cs.store(group="model", name="shallow_fusion_multi_ver2", node=AudioLanguageShallowMultiV2Config)
    cs.store(group="model", name="deep_fusion", node=AudioLanguageDeepConfig)
    cs.store(group="model", name="deep_fusion_pointer", node=AudioLanguageDeepPointerConfig)
    cs.store(group="model", name="deep_fusion_pointer_ver2", node=AudioLanguageDeepPointerV2Config)

    ## Audio Feature
    cs.store(group="audio_feature", name="raw", node=AudioFeatureConfig)
    cs.store(group="audio_feature", name="except", node=ExceptFeatureConfig)
    cs.store(group="audio_feature", name="mfcc", node=MfccFeatureConfig)
    cs.store(group="audio_feature", name="mel", node=MelFeatureConfig)
    cs.store(group="audio_feature", name="wav2vec", node=Wav2vecFeatureConfig)
    cs.store(group="audio_feature", name="vq_wav2vec", node=VQWav2vecFeatureConfig)
    cs.store(group="audio_feature", name="vq_token", node=VQTokenFeatureConfig)

    ## Audio Model

    ## Language Feature
    cs.store(group="language_feature", name="raw", node=LanguageFeatureConfig)
    cs.store(group="language_feature", name="hanbert", node=HanBertFeatureConfig)
    cs.store(group="language_feature", name="hantoken", node=HanTokenFeatureConfig)
    cs.store(group="language_feature", name="fasttext", node=FasttextFeatureConfig)

    ## split
    cs.store(group="split", name="three", node=SplitPreprocessConfig)

    ## label
    cs.store(group="label", name="five", node=FiveLabelPreprocessConfig)
    cs.store(group="label", name="seven", node=SevenLabelPreprocessConfig)
    cs.store(group="label", name="multi8", node=Multimodal8PreprocessConfig)
    cs.store(group="label", name="multi7", node=Multimodal7PreprocessConfig)
    cs.store(group="label", name="multi6", node=Multimodal6PreprocessConfig)
    cs.store(group="label", name="multi5", node=Multimodal5PreprocessConfig)
    cs.store(group="label", name="multi4", node=Multimodal4PreprocessConfig)
    cs.store(group="label", name="synthesis7", node=Synthesis7PreprocessConfig)

    ## Speech to Text API
    cs.store(group="api", name="google", node=GoogleConfig)
    cs.store(group="api", name="clova", node=ClovaConfig)
    cs.store(group="api", name="kakao", node=KaKaoConfig)


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


def load_data(cfg:DefaultConfig):
    data_path = cfg.data_path
    assert os.path.isfile(data_path), "Invalid data_path. check the data file [{}]".format(data_path)

    ## load csv file(official encoding for public data is 'cp949' or 'utf8')
    script_ext = Path(data_path).suffix.replace('.', '')
    if script_ext == 'csv':
        script_df = pd.read_csv(data_path, encoding='cp949', engine='python')
    elif script_ext == 'pkl':
        script_df = pd.read_pickle(data_path)
    else:
        raise ValueError("don't supported extion of script file [{}]".format(data_path))

    logging.info("load dataFrame from [{}]".format(data_path))
    return script_df

def save_data(df:pd.DataFrame ,cfg:DefaultConfig):
    save_path=cfg.save_path
    os.makedirs(Path(save_path).parent, exist_ok=True)

    df.to_pickle(save_path)
    logging.info("save dataFrame to [{}]".format(save_path))



class ResultWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, base_cfg, model_cfg, label_cfg, audio_cfg, language_cfg, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update({
            'audio_name': audio_cfg.name,
            'language_name': language_cfg.name,
            'label_name': label_cfg.name,
            'model_name': model_cfg.name,
        })
        self.writer.update(dict(base_cfg))
        self.writer.update(dict(model_cfg))
        self.writer.update(dict(label_cfg))
        self.writer.update(dict(audio_cfg))
        self.writer.update(dict(language_cfg))


        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()
        logging.info("save experiment info [{}]".format(self.dir))

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None