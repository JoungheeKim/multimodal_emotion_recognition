import sys

import torch
from torch import nn
from dataclasses import _MISSING_TYPE, dataclass, field
from .ad_lm_model import AudioLanguageModel, AudioLanguageModelConfig
from src.dataset.audio_dataset import AudioFeatureConfig
from src.dataset.language_dataset import LanguageFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig
from torch.nn import functional as F


@dataclass
class AudioLanguageShallowMultiV2Config(AudioLanguageModelConfig):
    name:str = 'shallow_fusion_multi_ver2'
    am_kernel_size: int = 3
    lm_kernel_size: int = 3
    hidden_dim: int = 20
    final_dropout: float = 0.1
    am_concat_proj:str=''
    lm_concat_proj:str=''
    am_last_hidden: bool = False
    lm_last_hidden: bool = False
    audio_multi_cross_step:int=1


class AudioLanguageShallowMultiV2Model(AudioLanguageModel):
    def __init__(self, model_cfg:AudioLanguageShallowMultiV2Config, ad_cfg:AudioFeatureConfig, lm_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(AudioLanguageShallowMultiV2Model, self).__init__(model_cfg, ad_cfg, lm_cfg, label_cfg)

        ## am type
        self.am_concat_proj = model_cfg.am_concat_proj
        am_concat_dim = self.ad_encoder_dim
        if self.am_concat_proj == 'conv':
            assert model_cfg.am_kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.am_kernel_size)
            am_padding = self._calculate_padding(model_cfg.am_kernel_size)
            self.audio_porj = nn.Conv1d(self.ad_encoder_dim, model_cfg.hidden_dim, kernel_size=model_cfg.am_kernel_size,
                                        padding=am_padding, bias=False)
            am_concat_dim = model_cfg.hidden_dim
        elif self.am_concat_proj == 'linear':
            self.audio_porj = nn.Linear(self.ad_encoder_dim, model_cfg.hidden_dim)
            am_concat_dim = model_cfg.hidden_dim
        self.am_proj = nn.Linear(am_concat_dim, self.num_class)


        ## lm type
        self.lm_concat_proj = model_cfg.lm_concat_proj
        lm_concat_dim = self.lm_encoder_dim
        if self.lm_concat_proj == 'conv':
            assert model_cfg.lm_kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.lm_kernel_size)
            lm_padding = self._calculate_padding(model_cfg.lm_kernel_size)
            self.text_proj = nn.Conv1d(self.lm_encoder_dim, model_cfg.hidden_dim,
                                        kernel_size=model_cfg.lm_kernel_size,
                                        padding=lm_padding, bias=False)
            lm_concat_dim = model_cfg.hidden_dim
        elif self.lm_concat_proj == 'linear':
            self.text_proj = nn.Linear(self.lm_encoder_dim, model_cfg.hidden_dim)
            lm_concat_dim = model_cfg.hidden_dim
        self.lm_proj = nn.Linear(lm_concat_dim, self.num_class)

        assert am_concat_dim==lm_concat_dim, 'concat dimention must be same {}=={}'.format(am_concat_dim, lm_concat_dim)

        self.final_dropout = nn.Dropout(model_cfg.final_dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(am_concat_dim + lm_concat_dim, am_concat_dim + lm_concat_dim)
        self.fc2 = nn.Linear(am_concat_dim + lm_concat_dim, am_concat_dim + lm_concat_dim)
        self.proj = nn.Linear(am_concat_dim + lm_concat_dim, self.num_class)

        self.am_last_hidden = model_cfg.am_last_hidden
        self.lm_last_hidden = model_cfg.lm_last_hidden

        self.audio_multi_cross_step = model_cfg.audio_multi_cross_step
        assert self.audio_multi_cross_step >=1




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
            'am_output': am_output,
            'lm_output': lm_output,
            'add_output': add_output,
            'add_am_output': add_am_output,
            'add_lm_output': add_lm_output,
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
        audio_encoding = self.final_dropout(audio_encoding)

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
        text_encoding = self.final_dropout(text_encoding)

        ## concat
        features = torch.cat([audio_encoding, text_encoding], dim=1)
        output = self.relu(self.fc1(features))
        output = self.final_dropout(output)
        output = self.fc2(output)
        output = features + output
        output = self.proj(output)

        am_output = self.am_proj(audio_encoding)
        lm_output = self.lm_proj(text_encoding)

        return output, am_output, lm_output

    def get_loss(self, net_output, labels):

        final_loss=0

        multi_portion = float(self.step/self.audio_multi_cross_step) if self.step <= self.audio_multi_cross_step else 1.0
        audio_portion = 1.0 - multi_portion

        lprobs = self.get_normalized_probs(
            net_output, log_probs=True, loss_type=self.loss_type, target='output'
        ).contiguous()
        fusion_loss = self.criterion(lprobs, labels).mean()
        final_loss = final_loss + (fusion_loss * multi_portion)

        am_lprobs = self.get_normalized_probs(
            net_output, log_probs=True, loss_type=self.loss_type, target='am_output'
        ).contiguous()
        am_loss = self.criterion(am_lprobs, labels).mean()
        final_loss = final_loss + (am_loss * audio_portion)

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

    def get_normalized_probs(self, net_output, log_probs=False, loss_type='bce', **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""

        target_name = kwargs.pop("target", 'output')

        logits = net_output[target_name]
        assert loss_type in ['bce', 'cross'], '여기서 박살나면 망함'

        if loss_type=='bce':
            return torch.sigmoid(logits.float())
        if loss_type=='cross':
            if log_probs:
                return F.log_softmax(logits.float(), dim=-1)
            else:
                return F.softmax(logits.float(), dim=-1)


