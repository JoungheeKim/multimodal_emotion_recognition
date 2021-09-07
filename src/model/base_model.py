import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
import logging

@dataclass
class ModelConfig:
    name:str


## 기본 메소드
class Pretrained_model(nn.Module):
    model_save_name = 'pretrained_model.bin'
    model_config_name = 'pretrained_config.bin'

    def __init__(self):
        super(Pretrained_model, self).__init__()
        self.step = 0
        #logging.info("### Load Model [{}]".format())

    def save_pretrained(self, save_path):
        # if not os.path.isdir(save_path):
        #     save_path = os.path.dirname(save_path)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, self.model_save_name))
        torch.save(self.cfg, os.path.join(save_path, self.model_config_name))
        logging.info('save files to [{}]'.format(save_path))

    @classmethod
    def from_pretrained(cls, pretrained_path):
        # if not os.path.isdir(pretrained_path):
        #     pretrained_path = os.path.dirname(pretrained_path)
        logging.info('load files from [{}]'.format(pretrained_path))
        cfg = torch.load(os.path.join(pretrained_path, cls.model_config_name))
        model = cls.load_with_config(cfg)
        model.load_state_dict(torch.load(os.path.join(pretrained_path, cls.model_save_name)))

        return model

    @classmethod
    def load_with_config(cls, cfg):
        raise NotImplementedError

    def set_config(self, cfg):
        self.cfg = cfg

    def set_step(self, step):
        self.step = step

    def get_normalized_probs(self, net_output, log_probs=False, loss_type='bce', **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["output"]
        assert loss_type in ['bce', 'cross'], '여기서 박살나면 망함'

        if loss_type=='bce':
            return torch.sigmoid(logits.float())
        if loss_type=='cross':
            if log_probs:
                return F.log_softmax(logits.float(), dim=-1)
            else:
                return F.softmax(logits.float(), dim=-1)


    def get_add_normalized_probs(self, net_output, log_probs=False, loss_type='bce', **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["add_output"]
        assert loss_type in ['bce', 'cross'], '여기서 박살나면 망함'

        if loss_type=='bce':
            return torch.sigmoid(logits.float())
        if loss_type=='cross':
            if log_probs:
                return F.log_softmax(logits.float(), dim=-1)
            else:
                return F.softmax(logits.float(), dim=-1)

    ## Flat Function
    def _masked_mean(self, tensor_x, mask=None, dim=0):
        if mask is None:
            return tensor_x.mean(dim=dim)
        mask = mask.unsqueeze(2)
        masked = tensor_x * mask  # Apply the mask using an element-wise multiply
        return masked.sum(dim=dim) / mask.sum(dim=dim)  # Find the average!

    ## Find Padding
    def _calculate_padding(self, kernel_size, dilation=1):
        padding = (dilation * (kernel_size - 1)) / 2
        return int(padding)