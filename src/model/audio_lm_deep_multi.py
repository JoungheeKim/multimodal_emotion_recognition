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
class AudioLanguageDeepMultiConfig(AudioLanguageModelConfig):
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

    multitune_start_step: int = 0
    multitune_end_step: int = sys.maxsize
    audiotune_start_step:int = 0
    audiotune_end_step: int = sys.maxsize
    texttune_start_step: int = 0
    texttune_end_step: int = sys.maxsize
    


class AudioLanguageDeepMultiModel(AudioLanguageModel):
    def __init__(self, model_cfg:AudioLanguageDeepMultiConfig, ad_cfg:AudioFeatureConfig, lm_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(AudioLanguageDeepMultiModel, self).__init__(model_cfg, ad_cfg, lm_cfg, label_cfg)

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
        self.am_proj = nn.Linear(am_hidden_dim, self.num_class)

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
        self.lm_proj = nn.Linear(lm_hidden_dim, self.num_class)

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

        self.multitune_start_step = model_cfg.multitune_start_step
        self.multitune_end_step = model_cfg.multitune_end_step

        self.texttune_start_step = model_cfg.texttune_start_step
        self.texttune_end_step = model_cfg.texttune_end_step

        self.audiotune_start_step = model_cfg.audiotune_start_step
        self.audiotune_end_step = model_cfg.audiotune_end_step



    def forward(self, input_ids, audio, token_type_ids=None, attention_mask=None, padding_mask=None, add_input_ids=None,
                add_token_type_ids=None, add_attention_mask=None, **kwargs):
        ## process foward
        output, am_output, lm_output = self.process(input_ids=input_ids,
                              audio=audio,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask,
                              padding_mask=padding_mask,
                              )

        add_output = None
        add_am_output=None
        add_lm_output=None
        if add_input_ids is not None and self.training:
            add_output, add_am_output, add_lm_output = self.process(input_ids=add_input_ids,
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
        only_audio_encoding = self.final_dropout(audio_encoding)

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
        only_text_encoding = self.final_dropout(text_encoding)


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

        return output, self.am_proj(only_audio_encoding), self.lm_proj(only_text_encoding)


    def get_loss(self, net_output, labels):

        final_loss=0

        if (self.multitune_start_step <= self.step and self.multitune_end_step >= self.step) or not self.training:
            lprobs = self.get_normalized_probs(
                net_output, log_probs=True, loss_type=self.loss_type, target='output'
            ).contiguous()
            fusion_loss = self.criterion(lprobs, labels).mean()
            final_loss = final_loss + fusion_loss

        if (self.audiotune_start_step <= self.step and self.audiotune_end_step >= self.step) and self.training:
            am_lprobs = self.get_normalized_probs(
                net_output, log_probs=True, loss_type=self.loss_type, target='am_output'
            ).contiguous()
            am_loss = self.criterion(am_lprobs, labels).mean()
            final_loss = final_loss + am_loss

        if (self.texttune_start_step <= self.step and self.texttune_end_step >= self.step) and self.training:
            lm_lprobs = self.get_normalized_probs(
                net_output, log_probs=True, loss_type=self.loss_type, target='lm_output'
            ).contiguous()
            lm_loss = self.criterion(lm_lprobs, labels).mean()
            final_loss = final_loss + lm_loss


        if self.training and net_output['add_output'] is not None:
            probs = self.get_normalized_probs(
                net_output, log_probs=False, loss_type=self.loss_type,
            ).contiguous()

            add_lprobs = self.get_add_normalized_probs(
                net_output, log_probs=True, loss_type=self.loss_type,
            ).contiguous()

            kl_fn = torch.nn.KLDivLoss(reduction="none")
            kl_loss = kl_fn(add_lprobs, probs).sum(dim=1)

            final_loss = final_loss + (kl_loss.mean() * self.coeff)

        return final_loss



