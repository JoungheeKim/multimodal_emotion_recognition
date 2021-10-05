from dataclasses import _MISSING_TYPE, dataclass, field
from omegaconf import II, MISSING
from typing import Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import torch

@dataclass
class BasicConfig:
    name: str
    column_name: str

@dataclass
class DefaultConfig:
    save_path:str = 'results/sentiment4_google_preprocessed.pkl'
    data_path: str = 'results/sentiment4_google_results.pkl'
    dump_path:str = ''
    experiments_path:str = '/code/gitRepo/sentiment_speech/experiments/experiment.csv'
    seed:int = 0

@dataclass
class TrainConfig(DefaultConfig):
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps:int = 2
    learning_rate:float = 1e-4
    wav2vec_learning_rate:float = -1.0
    bert_learning_rate:float = -1.0
    weight_decay:float = 0.01
    adam_epsilon:float = 1e-8
    max_grad_norm:float = 1.0
    num_train_epochs:int = 1
    max_steps:int  = -1
    warmup_steps: int = -1
    warmup_percent:float = 0.1
    logging_steps:int = 200
    pretrained_model:str = ''
    fp16:bool = False
    fp16_opt_level:str="O1"
    n_gpu:int=1
    local_rank:int=-1
    no_cuda:bool=False
    device:str='cuda'

@dataclass
class EvalConfig(DefaultConfig):
    eval_batch_size: int = 2
    pretrained_model:str = ''
    fp16:bool = False
    fp16_opt_level:str="O1"
    n_gpu:int=1
    local_rank:int=-1
    no_cuda:bool=False
    device:str='cuda'
    result_path:str=''


class PreprocessConfig(DefaultConfig):
    pass

class CallAPIConfig(DefaultConfig):
    pass
