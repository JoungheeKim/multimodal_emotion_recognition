import io
import os
import pandas as pd
from pathlib import Path
import librosa
import torchaudio
import torch
from dataclasses import _MISSING_TYPE, dataclass, field
from torch import nn
from .audio_model import AudioModelConfig, AudioModel
from src.dataset.audio_dataset import AudioFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig
import torch.nn.functional as F

@dataclass
class Wav2KeywordConfig(AudioModelConfig):
    name:str= "wav2keyword"
    hidden_dim:int=256


class Wav2Keyword(AudioModel):
    def __init__(self, model_cfg:Wav2KeywordConfig, feature_cfg:AudioFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(Wav2Keyword, self).__init__(model_cfg, feature_cfg, label_cfg)
        hidden_dim = model_cfg.hidden_dim
        encoder_dim = self.encoder_dim

        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_dim, hidden_dim, 25, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, self.num_class, 1)
        )

    def forward(self, audio, padding_mask=None, **kwargs):

        if hasattr(self, 'w2v_encoder'):
            input_args = {
                "source": audio,
                "padding_mask": padding_mask,
                "mask": padding_mask,
            }
            audio, padding_mask = self.model.extract_features(**input_args)

        print('input shape', audio.shape)

        b ,t ,c = audio.shape
        audio = audio.reshape(b ,c ,t)
        output = self.decoder(audio).squeeze()

        print('output shape', output.shape)

        return {
            'output' : output,
            'padding_mask' : padding_mask,
        }

    def get_logits(self, net_output):
        logits = net_output["output"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def get_normalized_probs(self, net_output, log_probs=False, loss_type='bce', **kwargs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["output"]
        if loss_type == 'bce':
            return torch.sigmoid(logits.float())
        else:  ## Cross Entropy
            if log_probs:
                return F.log_softmax(logits.float(), dim=-1)
            else:
                return F.softmax(logits.float(), dim=-1)