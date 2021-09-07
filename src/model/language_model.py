import torch
from torch import nn
import os
import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from src.dataset.language_dataset import LanguageFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig
import fairseq
from .base_model import Pretrained_model, ModelConfig
from transformers import BertModel

from collections import OrderedDict

LOSS_MAPPING = OrderedDict(
    [
        # Add configs here
        ("bce", torch.nn.BCELoss()),
        ('cross', torch.nn.NLLLoss()),
    ]
)

@dataclass
class LanguageModelConfig(ModelConfig):
    name:str
    bert_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/HanBert-54kN-torch'
    use_bert:bool = False
    finetune_bert:bool = False
    last_hidden: bool = False

    coeff: float = 1.0


class LanguageModel(Pretrained_model):
    def __init__(self, model_cfg:LanguageModelConfig, feature_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(LanguageModel, self).__init__()
        ## set model base attribute
        self.name = model_cfg.name
        self.bert_path = model_cfg.bert_path
        self.use_bert = model_cfg.use_bert
        self.finetune_bert = model_cfg.finetune_bert

        ## set feature base attribute
        self.feature_dim = feature_cfg.feature_dim

        ## set label base attribtue
        self.num_class = len(eval(label_cfg.selected_class))

        if self.use_bert:
            model = BertModel.from_pretrained(self.bert_path)
            self.bert_encoder=model
            encoder_dim = model.config.hidden_size
        else:
            encoder_dim = self.feature_dim

        self.encoder_dim = encoder_dim

        ## set Criterion
        self.criterion = LOSS_MAPPING[label_cfg.loss_type]
        self.loss_type = label_cfg.loss_type

        ## Coeff
        self.coeff = model_cfg.coeff

    def bert_encoding(self, input_ids, token_type_ids=None, attention_mask=None, **kwargs):
        last_hidden = kwargs.pop('last_hidden', False)

        if hasattr(self, 'bert_encoder'):
            if self.finetune_bert:
                outputs = self.bert_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            else:
                with torch.no_grad():
                    self.bert_encoder.eval()
                    outputs = self.bert_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            if last_hidden:
                outputs=outputs[1]
            else:
                outputs=outputs[0]
        else:
            outputs = input_ids
        return outputs


    @classmethod
    def load_with_config(cls, cfg):
        model_cfg = cfg.model
        audio_feature_cfg = cfg.audio_feature
        label_cfg = cfg.label

        model = cls(model_cfg, audio_feature_cfg, label_cfg)
        model.set_config(cfg)
        return model


    def get_loss(self, net_output, labels):
        lprobs = self.get_normalized_probs(
            net_output, log_probs=True, loss_type=self.loss_type,
        ).contiguous()

        final_loss = self.criterion(lprobs, labels).mean()

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
