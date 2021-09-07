import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from .base_preprocess import BaseProcessor, BasicProcessConfig

@dataclass
class SplitPreprocessConfig(BasicProcessConfig):
    name:str="split"
    test_ratio:float = 0.1
    column_name:str = 'split'

class SplitProcessor(BaseProcessor):
    def __init__(self, cfg:SplitPreprocessConfig):
        super(SplitProcessor, self).__init__(cfg)
        self.test_ratio = cfg.test_ratio
        self.column_name = cfg.column_name
        self.stratify_name = cfg.stratify_name

        self.train_name = 'train'
        self.valid_name = 'valid'
        self.test_name = 'test'

    def convert_data(self, script_df):
        train_df, test_df = train_test_split(script_df, test_size=self.test_ratio * 2, stratify=script_df[self.stratify_name],
                                             random_state=42)
        valid_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df[self.stratify_name], random_state=42)
        train_df.loc[:, self.column_name] = self.train_name
        valid_df.loc[:, self.column_name] = self.valid_name
        test_df.loc[:, self.column_name] = self.test_name

        script_df = pd.concat([train_df, valid_df, test_df])

        return script_df

    def get_split_data(self, script_df):
        train_df = script_df[script_df[self.column_name]==self.train_name].copy()
        valid_df = script_df[script_df[self.column_name]==self.valid_name].copy()
        test_df = script_df[script_df[self.column_name]==self.test_name].copy()

        return train_df, valid_df, test_df

