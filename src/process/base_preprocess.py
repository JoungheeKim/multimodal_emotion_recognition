
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import yaml
from types import SimpleNamespace
import json
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from src.configs import BasicConfig

@dataclass
class BasicProcessConfig(BasicConfig):
    stratify_name:str='label_name'


class BaseProcessor:
    def __init__(self, cfg:BasicProcessConfig):
        return

    def convert_data(self, script_df):
        raise NotImplementedError
