import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import _MISSING_TYPE, dataclass, field
from .language_model import LanguageModelConfig, LanguageModel
from src.dataset.language_dataset import LanguageFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig

@dataclass
class LMTransformerConfig(LanguageModelConfig):
    name:str = 'lm_transformer'
    num_layers: int = 4
    hidden_dim: int = 20
    head_num: int = 4
    final_dropout:float=0.1
    last_hidden: bool = False
    max_pool: bool = False
    kernel_size: int = 3
    proj_layer:str=''
        


class LMTransformerModel(LanguageModel):
    def __init__(self, model_cfg:LMTransformerConfig, feature_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(LMTransformerModel, self).__init__(model_cfg, feature_cfg, label_cfg)

        hidden_dim = self.encoder_dim
        self.proj_layer = model_cfg.proj_layer
        if self.proj_layer == 'conv':
            assert model_cfg.kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.kernel_size)
            padding = self._calculate_padding(model_cfg.kernel_size)
            self.text_proj = nn.Conv1d(self.lm_encoder_dim, model_cfg.hidden_dim,
                                       kernel_size=model_cfg.kernel_size,
                                       padding=padding, bias=False)
            hidden_dim = model_cfg.hidden_dim
        elif self.proj_layer == 'linear':
            self.text_proj = nn.Linear(self.encoder_dim, model_cfg.hidden_dim)
            hidden_dim = model_cfg.hidden_dim


        self.layers = nn.ModuleList([])
        for layer in range(model_cfg.num_layers):
            new_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=model_cfg.head_num)
            self.layers.append(new_layer)

        #self.encoder = nn.Conv1d(768, args.hidden_dim, kernel_size=3, padding=1, bias=False)  # 5
        self.final_dropout = nn.Dropout(model_cfg.final_dropout)
        self.proj = nn.Linear(hidden_dim, self.num_class)

        self.last_hidden=model_cfg.last_hidden
        self.max_pool = model_cfg.max_pool


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, add_input_ids=None, add_token_type_ids=None, add_attention_mask=None, **kwargs):
        output = self.process(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        add_output=None
        if add_input_ids is not None and self.training:
            add_output = self.process(add_input_ids, token_type_ids=add_token_type_ids, attention_mask=add_attention_mask)

        return {
            'output': output,
            'add_output' : add_output,
        }

    def process(self, input_ids, token_type_ids=None, attention_mask=None, **kwargs):

        #print("input_ids", input_ids.shape)
        

        outputs = self.bert_encoding(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if hasattr(self, 'text_proj'):
            if type(self.text_proj) == nn.Conv1d:
                # for conv, (B, L, D) => (B, D, L)
                outputs = outputs.transpose(1, 2)
                # (B, d, L) => (B, L, d)
                outputs = self.text_proj(outputs).transpose(1, 2)
            else:
                outputs = self.text_proj(outputs)

        ## Calculate Padding
        text_pad_true_mask=None
        text_pad_false_mask=None
        if attention_mask is not None:
            if len(attention_mask.shape) > 2:
                attention_mask = attention_mask[:, :, 0]
            text_pad_true_mask = (attention_mask == 0) if attention_mask is not None else None
            text_pad_false_mask = attention_mask.bool() if attention_mask is not None else None

        # for transformer, (B, L, D) => (L, B, D)
        outputs = outputs.transpose(0, 1)

        #print("outputs0", outputs.shape)

        for layer in self.layers:
            outputs = layer(outputs, src_key_padding_mask=text_pad_true_mask)
            #outputs = layer(outputs, src_key_padding_mask=text_pad_false_mask)

        # for transformer, (L, B, D) => (B, L, D)
        outputs = outputs.transpose(0, 1)


        #print("text_pad_false_mask", text_pad_false_mask)
        #print("text_pad_true_mask", text_pad_true_mask)

        if not self.last_hidden:
            outputs = self._masked_mean(outputs, text_pad_false_mask, dim=1)
            #outputs = self._masked_mean(outputs, text_pad_true_mask, dim=1)
        else:
            # (bat, dim)
            outputs = outputs[:,0,:]

        #print("outputs1", outputs.shape)

        outputs = self.final_dropout(outputs)
        outputs = self.proj(outputs)

        #print("outputs2", outputs.shape)

        return outputs

    def get_logits(self, net_output):
        logits = net_output["output"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float('-inf')

        return logits

    
