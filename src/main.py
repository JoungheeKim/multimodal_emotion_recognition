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

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from tqdm import tqdm, trange
import pandas as pd
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torch.utils.data import (
    DataLoader, Dataset
)
import random
import numpy as np
import pickle
import torch
import os
#from src.configs import add_defaults
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from hydra.core.config_store import ConfigStore
from audio.base_audio import MfccConfig
from model.conformer import ConformerConfig

@dataclass
class TrainConfig:
    seed:int = 0
    what:int = 20
    please:int = 30

def init():
    cs = ConfigStore.instance()
    cs.store(group="train", name='train', node=TrainConfig)
    cs.store(group="audio_model", name="conformer", node=ConformerConfig)
    cs.store(group="audio_feature", name="mfcc", node=MfccConfig)
    cs.store(group="audio_feature", name="raw", node=MfccConfig)
    cs.store(group="audio_feature", name="wav2vec", node=MfccConfig)

    cs.store(group="language_feature", name="raw", node=MfccConfig)
    cs.store(group="language_feature", name="bert", node=MfccConfig)

    cs.store(group="preprocess", name="kakao", node=MfccConfig)
    cs.store(group="preprocess", name="google", node=MfccConfig)
    cs.store(group="preprocess", name="naver", node=MfccConfig)


@hydra.main(config_path=os.path.join("..", "configs"), config_name="default")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))







if __name__ == "__main__":
    init()
    main()

