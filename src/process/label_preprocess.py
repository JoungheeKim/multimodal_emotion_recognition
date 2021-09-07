import io
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from .base_preprocess import BaseProcessor, BasicProcessConfig
import numpy as np
import logging
from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class LabelPreprocessConfig(BasicProcessConfig):
    name:str='labels'
    column_name: str = 'labels'
    selected_columns:str = "['1번 감정', '2번 감정', '3번 감정', '4번 감정', '5번 감정']"
    selected_class:str = "['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']"
    loss_type:str = 'bce'

@dataclass
class FiveLabelPreprocessConfig(LabelPreprocessConfig):
    name = '5label'
    selected_class:str = "['Angry', 'Disgust', 'Fear', 'Neutral', 'Sadness']"

@dataclass
class SevenLabelPreprocessConfig(LabelPreprocessConfig):
    name = '7label'
    selected_class:str = "['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']"


@dataclass
class Multimodal8PreprocessConfig(LabelPreprocessConfig):
    name = 'multimodal8'
    selected_columns='[]'
    selected_class:str = "['dislike', 'happy', 'surprise', 'neutral', 'sad', 'angry', 'fear', 'contempt']"

@dataclass
class Multimodal7PreprocessConfig(LabelPreprocessConfig):
    name = 'multimodal7'
    selected_columns = '[]'
    selected_class:str = "['dislike', 'happy', 'surprise', 'neutral', 'sad', 'angry', 'fear']"

@dataclass
class Multimodal6PreprocessConfig(LabelPreprocessConfig):
    name = 'multimodal6'
    selected_columns = '[]'
    selected_class:str = "['dislike', 'happy', 'surprise', 'neutral', 'sad', 'angry']"

class Multimodal5PreprocessConfig(LabelPreprocessConfig):
    name = 'multimodal5'
    selected_columns = '[]'
    selected_class:str = "['dislike', 'happy', 'surprise', 'neutral', 'sad']"

@dataclass
class Multimodal4PreprocessConfig(LabelPreprocessConfig):
    name = 'multimodal4'
    selected_columns = '[]'
    selected_class:str = "['dislike', 'happy', 'surprise', 'neutral']"

@dataclass
class Synthesis7PreprocessConfig(LabelPreprocessConfig):
    name = 'synthesis7'
    selected_columns = '[]'
    selected_class:str = "['ang', 'dis', 'fea', 'hap', 'neu', 'sad', 'sur']"


class LabelProcessor(BaseProcessor):
    def __init__(self, cfg:LabelPreprocessConfig):
        super(LabelProcessor, self).__init__(cfg)
        self.selected_columns=eval(cfg.selected_columns)
        self.selected_class=eval(cfg.selected_class)
        self.column_name=cfg.column_name
        self.stratify_name = cfg.stratify_name
        self.loss_type = cfg.loss_type.lower()
        assert self.loss_type in ['bce', 'cross']

    def convert_data(self, script_df):
        if len(self.selected_columns) > 0:
            ## get Emotions
            emotion_lists = script_df[self.selected_columns].apply(pd.value_counts).T.columns

            before_num = len(script_df)
            ## get major emotion
            mask = script_df[self.selected_columns].apply(pd.value_counts, axis=1).max(axis=1) > 2
            script_df = script_df[mask]
            after_num = len(script_df)
            logging.info('befor [{}] after [{}]'.format(before_num, after_num))

            major_label_index = np.nanargmax(
                script_df[self.selected_columns].apply(pd.value_counts, axis=1)[emotion_lists].values, axis=1)

            ## set label_name
            script_df.loc[:, self.stratify_name] = emotion_lists[major_label_index]

        ## 추가부분
        mask = script_df[self.stratify_name].isin(self.selected_class)
        script_df = script_df[mask]

        logging.info("convert label to selected class")
        selected_dict = {name: idx for idx, name in enumerate(self.selected_class)}
        script_df[self.column_name] = script_df[self.stratify_name].map(selected_dict)

        return script_df

    def get_data(self, script_df):
        ## extract selected class data

        labels = script_df[self.column_name].values.tolist()
        if self.loss_type =='bce':
            logging.info('load Byte Cross Entropy Labels')
            targets = np.array(labels)
            labels = np.eye(len(self.selected_class))[targets]

        return labels


