import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
from collections import OrderedDict
from conformer import ConformerBlock
from dataclasses import _MISSING_TYPE, dataclass, field
from .audio_model import AudioModelConfig, AudioModel
from src.dataset.audio_dataset import AudioFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig

@dataclass
class ConformerConfig(AudioModelConfig):
    name:str = 'conformer'
    kernel_size: int = 32
    hidden_dim: int = 80
    head_dim: int = 10
    head_num: int = 4
    ff_mult: int = 4
    num_layers: int = 4
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    conv_dropout: float = 0.1
    final_dropout: float = 0.1
    last_hidden:bool=False
    max_pool: bool = False

class Conformer(AudioModel):
    def __init__(self, model_cfg:ConformerConfig, feature_cfg:AudioFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(Conformer, self).__init__(model_cfg, feature_cfg, label_cfg)

        def calculate_padding(kernel_size, dilation=1):
            padding = (dilation * (kernel_size - 1)) / 2
            return int(padding)

        assert model_cfg.kernel_size % 2 == 1, "kernel은 항상 홀수여야 함 다시 확인하세요".format(model_cfg.kernel_size)
        padding = calculate_padding(model_cfg.kernel_size)

        self.encoder = nn.Conv1d(self.encoder_dim, model_cfg.hidden_dim, kernel_size=model_cfg.kernel_size, padding=padding, bias=False)

        self.layers = nn.ModuleList([])
        for layer in range(model_cfg.num_layers):
            new_layer = ConformerBlock(
                dim=model_cfg.hidden_dim,
                dim_head=model_cfg.head_dim,
                heads=model_cfg.head_num,
                ff_mult=model_cfg.ff_mult,
                conv_expansion_factor=2,
                conv_kernel_size=model_cfg.kernel_size,
                attn_dropout=model_cfg.attn_dropout,
                ff_dropout=model_cfg.ff_dropout,
                conv_dropout=model_cfg.conv_dropout,
            )
            self.layers.append(new_layer)

        #self.encoder = nn.Conv1d(768, args.hidden_dim, kernel_size=3, padding=1, bias=False)  # 5
        self.final_dropout = nn.Dropout(model_cfg.final_dropout)
        self.proj = nn.Linear(model_cfg.hidden_dim, self.num_class)

        self.last_hidden=model_cfg.last_hidden
        self.max_pool = model_cfg.max_pool

    def forward(self, audio, padding_mask=None, **kwargs):
        self.w2v_encoder.eval()
        audio, padding_mask = self.w2v_encoding(audio, padding_mask)

        # for conv, (B, L, D) => (B, D, L)
        x = audio.transpose(1, 2)

        # (B, d, L) => (B, L, d)
        x = self.encoder(x).transpose(1, 2)

        audio_pad_true_mask = padding_mask.bool() if padding_mask is not None else None
        audio_pad_false_mask = (padding_mask==0).bool() if padding_mask is not None else None

        for layer in self.layers:
            x = layer(x)

        if not self.last_hidden:
            def masked_mean(tensor_x, mask=None, dim=0):
                if mask is None:
                    return tensor_x.mean(dim=dim)
                mask = mask.unsqueeze(2)
                masked = tensor_x * mask  # Apply the mask using an element-wise multiply
                return masked.sum(dim=dim) / mask.sum(dim=dim)  # Find the average!
            output = masked_mean(x, audio_pad_false_mask, dim=1)
        else:
            # (bat, dim)
            output = x[:,-1,:]

        output = self.final_dropout(output)
        output = self.proj(output)

        return {
            'output': output,
            'padding_mask': padding_mask,
        }

    def get_logits(self, net_output):
        logits = net_output["output"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float('-inf')

        return logits


