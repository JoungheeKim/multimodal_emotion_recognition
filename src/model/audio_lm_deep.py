import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import _MISSING_TYPE, dataclass, field
from .ad_lm_model import AudioLanguageModel, AudioLanguageModelConfig
from .modules import CrossmodalTransformer
from src.dataset.audio_dataset import AudioFeatureConfig
from src.dataset.language_dataset import LanguageFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig
from fairseq.modules import SinusoidalPositionalEmbedding
from .modules import CrossmodalTransformer

@dataclass
class AudioLanguageDeepConfig(AudioLanguageModelConfig):
    name:str = 'deep_fusion'
    am_kernel_size: int = 3
    lm_kernel_size: int = 3
    final_dropout: float = 0.1
    am_concat_proj:str=''
    lm_concat_proj:str=''
    am_last_hidden: bool = False
    lm_last_hidden: bool = False
    fusion_last_hidden: bool = False

    num_layers: int = 4
    head_num: int = 4
    hidden_dim: int = 20


class AudioLanguageDeepModel(AudioLanguageModel):
    def __init__(self, model_cfg:AudioLanguageDeepConfig, ad_cfg:AudioFeatureConfig, lm_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(AudioLanguageDeepModel, self).__init__(model_cfg, ad_cfg, lm_cfg, label_cfg)

        ## am type
        self.am_concat_proj = model_cfg.am_concat_proj
        am_hidden_dim = self.ad_encoder_dim
        if self.am_concat_proj == 'conv':
            assert model_cfg.am_kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.am_kernel_size)
            am_padding = self._calculate_padding(model_cfg.am_kernel_size)
            self.audio_porj = nn.Conv1d(self.ad_encoder_dim, model_cfg.hidden_dim, kernel_size=model_cfg.am_kernel_size,
                                        padding=am_padding, bias=False)
            am_hidden_dim = model_cfg.hidden_dim
        elif self.am_concat_proj == 'linear':
            self.audio_porj = nn.Linear(self.ad_encoder_dim, model_cfg.hidden_dim)
            am_hidden_dim = model_cfg.hidden_dim

        ## lm type
        self.lm_concat_proj = model_cfg.lm_concat_proj
        lm_hidden_dim = self.lm_encoder_dim
        if self.lm_concat_proj == 'conv':
            assert model_cfg.lm_kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.lm_kernel_size)
            lm_padding = self._calculate_padding(model_cfg.lm_kernel_size)
            self.text_proj = nn.Conv1d(self.lm_encoder_dim, model_cfg.hidden_dim,
                                        kernel_size=model_cfg.lm_kernel_size,
                                        padding=lm_padding, bias=False)
            lm_hidden_dim = model_cfg.hidden_dim
        elif self.lm_concat_proj == 'linear':
            self.text_proj = nn.Linear(self.lm_encoder_dim, model_cfg.hidden_dim)
            lm_hidden_dim = model_cfg.hidden_dim

        assert am_hidden_dim==lm_hidden_dim, 'concat dimention must be same {}=={}'.format(am_hidden_dim, lm_hidden_dim)
        assert am_hidden_dim==model_cfg.hidden_dim, 'concat dimention must be same {}=={}'.format(am_hidden_dim, model_cfg.hidden_dim)

        self.layers = nn.ModuleList([])

        for layer in range(model_cfg.num_layers):
            new_layer = nn.TransformerEncoderLayer(d_model=model_cfg.hidden_dim, nhead=model_cfg.head_num)
            self.layers.append(new_layer)

        self.final_dropout = nn.Dropout(model_cfg.final_dropout)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(model_cfg.hidden_dim, self.num_class)

        self.am_last_hidden = model_cfg.am_last_hidden
        self.lm_last_hidden = model_cfg.lm_last_hidden
        self.fusion_last_hidden = model_cfg.fusion_last_hidden


    def forward(self, input_ids, audio, token_type_ids=None, attention_mask=None, padding_mask=None, add_input_ids=None,
                add_token_type_ids=None, add_attention_mask=None, **kwargs):
        ## process foward
        output = self.process(input_ids=input_ids,
                              audio=audio,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask,
                              padding_mask=padding_mask,
                              )

        add_output=None
        if add_input_ids is not None and self.training:
            add_output = self.process(input_ids=add_input_ids,
                                      audio=audio,
                                      token_type_ids=add_token_type_ids,
                                      attention_mask=add_attention_mask,
                                      padding_mask=padding_mask,
                                      )

        return {
            'output': output,
            'add_output': add_output,
        }

    def process(self, input_ids, audio, token_type_ids=None, attention_mask=None, padding_mask=None, **kwargs):

        ## Audio
        audio_encoding, padding_mask = self.w2v_encoding(audio, padding_mask)
        if hasattr(self, 'audio_porj'):
            if type(self.audio_porj) == nn.Conv1d:
                # for conv, (B, L, D) => (B, D, L)
                audio_encoding = audio_encoding.transpose(1, 2)
                # (B, d, L) => (B, L, d)
                audio_encoding = self.audio_porj(audio_encoding).transpose(1, 2)
            else:
                audio_encoding = self.audio_porj(audio_encoding)

        ## Calculate Padding
        if len(padding_mask.shape) > 2:
            padding_mask = padding_mask[:, :, 0]
        audio_pad_true_mask = padding_mask.bool() if padding_mask is not None else None
        audio_pad_false_mask = (padding_mask == 0) if padding_mask is not None else None
        if not self.am_last_hidden:
            audio_encoding = self._masked_mean(audio_encoding, audio_pad_false_mask, dim=1)
        else:
            audio_encoding = audio_encoding[:,0,:]
        
        ## Text
        text_encoding = self.bert_encoding(input_ids=input_ids, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)
        if hasattr(self, 'text_proj'):
            if type(self.text_proj) == nn.Conv1d:
                # for conv, (B, L, D) => (B, D, L)
                text_encoding = text_encoding.transpose(1, 2)
                # (B, d, L) => (B, L, d)
                text_encoding = self.text_proj(text_encoding).transpose(1, 2)
            else:
                text_encoding = self.text_proj(text_encoding)

        ## Calculate Padding
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, 0]
        text_pad_true_mask = (attention_mask == 0) if attention_mask is not None else None
        text_pad_false_mask = attention_mask.bool() if attention_mask is not None else None
        if not self.lm_last_hidden:
            text_encoding = self._masked_mean(text_encoding, text_pad_false_mask, dim=1)
        else:
            text_encoding = text_encoding[:,0,:]
        
        audio_encoding = audio_encoding.unsqueeze(1)
        text_encoding = text_encoding.unsqueeze(1)
        features = torch.cat([audio_encoding, text_encoding], dim=1)

        # for conv, (B, L, D) => (L, B, D)
        features = features.transpose(0, 1)

        for layer in self.layers:
            features = layer(features)

        # for conv, (L, B, D) => (B, L, D)
        features = features.transpose(0, 1)

        if not self.fusion_last_hidden:
            features = self._masked_mean(features, None, dim=1)
        else:
            features = features[:,0,:]

        ## concat
        #output = self.final_dropout(output)
        output = self.proj(features)

        return output


    



