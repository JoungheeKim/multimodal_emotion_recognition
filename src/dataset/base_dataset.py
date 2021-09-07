from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import yaml
import json
from src.configs import BasicConfig
from dataclasses import _MISSING_TYPE, dataclass, field

## All Process is done with Pandas Framework

@dataclass
class FeatureConfig(BasicConfig):
    feature_dim:int

class BaseProcessor:
    def __init__(self, config):
        return

    def convert_data(self, script_df):
        raise NotImplementedError

    def get_data(self, script_df):
        raise NotImplementedError


    def load_file(self, temp_path):
        ## load csv file(official encoding for public data is 'cp949' or 'utf8')
        ext = Path(temp_path).suffix.replace('.', '')
        if ext == 'csv':
            df = pd.read_csv(temp_path, encoding='cp949', engine='python')
        elif ext == 'pkl':
            df = pd.read_pickle(temp_path)
        else:
            raise ValueError("don't supported extion of script file [{}]".format(temp_path))
        return df

    @classmethod
    def from_config(cls, config):
        obj = cls(config)
        return obj

