import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import _MISSING_TYPE, dataclass, field
from .audio_model import AudioModel, AudioModelConfig
from src.dataset.audio_dataset import AudioFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig

@dataclass
class AudioTransformerConfig(AudioModelConfig):
    name:str = 'audio_transformer'
    num_layers: int = 4
    hidden_dim: int = 20
    head_num: int = 4
    final_dropout:float=0.1
    last_hidden: bool = False
    max_pool: bool = False
    kernel_size: int = 3
    proj_layer:str=''


class AudioTransformerModel(AudioModel):
    def __init__(self, model_cfg:AudioTransformerConfig, feature_cfg:AudioFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(AudioTransformerModel, self).__init__(model_cfg, feature_cfg, label_cfg)

        def calculate_padding(kernel_size, dilation=1):
            padding = (dilation * (kernel_size - 1)) / 2
            return int(padding)

        hidden_dim = self.encoder_dim
        self.proj_layer = model_cfg.proj_layer
        if self.proj_layer == 'conv':
            assert model_cfg.kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.kernel_size)
            padding = self._calculate_padding(model_cfg.kernel_size)
            self.audio_porj = nn.Conv1d(self.encoder_dim, model_cfg.hidden_dim,
                                       kernel_size=model_cfg.kernel_size,
                                       padding=padding, bias=False)
            hidden_dim = model_cfg.hidden_dim
        elif self.proj_layer == 'linear':
            self.audio_porj = nn.Linear(self.encoder_dim, model_cfg.hidden_dim)
            hidden_dim = model_cfg.hidden_dim
        
        self.layers = nn.ModuleList([])

        #print("hidden_dim", hidden_dim)
        #print("encoder_dim", self.encoder_dim)

        #self.encoder_norm = LayerNorm(self.encoder_dim)
        for layer in range(model_cfg.num_layers):
            new_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=model_cfg.head_num)
            self.layers.append(new_layer)

        #self.encoder = nn.Conv1d(768, args.hidden_dim, kernel_size=3, padding=1, bias=False)  # 5
        self.final_dropout = nn.Dropout(model_cfg.final_dropout)
        self.proj = nn.Linear(hidden_dim, self.num_class)

        self.last_hidden=model_cfg.last_hidden
        self.max_pool = model_cfg.max_pool

    def forward(self, audio, padding_mask=None, **kwargs):

        ## Audio
        outputs, padding_mask = self.w2v_encoding(audio, padding_mask)

        if hasattr(self, 'audio_porj'):
            if type(self.audio_porj) == nn.Conv1d:
                # for conv, (B, L, D) => (B, D, L)
                outputs = outputs.transpose(1, 2)
                # (B, d, L) => (B, L, d)
                outputs = self.audio_porj(outputs).transpose(1, 2)
            else:
                outputs = self.audio_porj(outputs)

        ## Calculate Padding
        if len(padding_mask.shape) > 2:
            padding_mask = padding_mask[:, :, 0]
        audio_pad_true_mask = padding_mask.bool() if padding_mask is not None else None
        audio_pad_false_mask = (padding_mask == 0) if padding_mask is not None else None
        
        # for conv, (B, L, D) => (L, B, D)
        outputs = outputs.transpose(0, 1)

        for layer in self.layers:
            outputs = layer(outputs, src_key_padding_mask=audio_pad_true_mask)

        # for conv, (L, B, D) => (B, L, D)
        outputs = outputs.transpose(0, 1)

        #print('audio_pad_false_mask', audio_pad_false_mask)
        #print('audio_pad_true_mask', audio_pad_true_mask)


        if not self.last_hidden:
            outputs = self._masked_mean(outputs, audio_pad_false_mask, dim=1)
        else:
            # (bat, dim)
            outputs = outputs[:,0,:]

        #print("outputs1", outputs.shape)

        outputs = self.final_dropout(outputs)
        outputs = self.proj(outputs)

        return {
            'output': outputs,
        }

    def get_logits(self, net_output):
        logits = net_output["output"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float('-inf')

        return logits

