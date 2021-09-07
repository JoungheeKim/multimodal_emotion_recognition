import torch
from torch import nn
import os
import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional
from src.dataset.audio_dataset import AudioFeatureConfig
from src.dataset.language_dataset import LanguageFeatureConfig
from src.process.label_preprocess import LabelPreprocessConfig
import fairseq
from .base_model import Pretrained_model, ModelConfig

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import BertTokenizerFast, BertModel
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch import Tensor
from typing import Optional, Any
from collections import OrderedDict

LOSS_MAPPING = OrderedDict(
    [
        # Add configs here
        ("bce", torch.nn.BCELoss()),
        ('cross', torch.nn.NLLLoss()),
    ]
)


@dataclass
class AudioLanguageModelConfig(ModelConfig):
    name:str
    wav2vec_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/wav2vec_small.pt'
    roberta_path: str = '/code/gitRepo/sentiment_speech/pretrained_model/english/roberta'
    wav2vec_name: str = 'wav2vec'
    use_wav2vec:bool = False
    finetune_wav2vec:bool = False
    bert_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/HanBert-54kN-torch'
    use_bert:bool = False
    finetune_bert:bool = False
    last_hidden: bool = False
    coeff: float = 1.0


class AudioLanguageModel(Pretrained_model):
    def __init__(self, model_cfg:AudioLanguageModelConfig, ad_cfg:AudioFeatureConfig, lm_cfg:LanguageFeatureConfig, label_cfg:LabelPreprocessConfig):
        super(AudioLanguageModel, self).__init__()
        ## set model base attribute
        self.name = model_cfg.name
        self.wav2vec_path = model_cfg.wav2vec_path
        self.roberta_path = model_cfg.roberta_path
        self.use_wav2vec = model_cfg.use_wav2vec
        self.finetune_wav2vec = model_cfg.finetune_wav2vec
        self.wav2vec_name = model_cfg.wav2vec_name

        ## set feature base attribute
        self.ad_feature_dim = ad_cfg.feature_dim

        ## set label base attribtue
        self.num_class = len(eval(label_cfg.selected_class))

        if self.use_wav2vec:
            if model_cfg.wav2vec_name == 'wav2vec':
                assert os.path.isfile(self.wav2vec_path), "No wav2vec file"
                assert self.ad_feature_dim == 1, 'if you want to use wav2vec, need to use audio feature config which has feature_dim==1'
                state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(self.wav2vec_path, None)
                w2v_args = state.get("cfg", None)
                task = fairseq.tasks.audio_pretraining.AudioPretrainingTask.setup_task(w2v_args.task)
                model = task.build_model(w2v_args.model)
                model.load_state_dict(state["model"], strict=True)
                model.remove_pretraining_modules()
                self.w2v_encoder = model
                ad_encoder_dim = w2v_args.model.encoder_embed_dim
            elif model_cfg.wav2vec_name == 'vq_wav2vec':
                assert os.path.isdir(self.roberta_path), "No Roberta directory [{}]".format(self.roberta_path)
                assert self.ad_feature_dim == 1, 'if you want to use wav2vec, need to use audio feature config which has feature_dim==1'
                ## load roberta
                model = fairseq.models.roberta.RobertaModel.from_pretrained(self.roberta_path)
                self.w2v_encoder = model
                w2v_args = model.cfg
                ad_encoder_dim = w2v_args.model.encoder_embed_dim

            else:
                raise ValueError('not supported wav2vec architecture. check again. [{}]'.format(model_cfg.wav2vec_name))
        else:
            ad_encoder_dim = self.ad_feature_dim

        self.ad_encoder_dim = ad_encoder_dim

        self.bert_path = model_cfg.bert_path
        self.use_bert = model_cfg.use_bert
        self.finetune_bert = model_cfg.finetune_bert

        ## set feature base attribute
        self.lm_feature_dim = lm_cfg.feature_dim

        ## set label base attribtue
        self.num_class = len(eval(label_cfg.selected_class))

        if self.use_bert:
            model = BertModel.from_pretrained(self.bert_path)
            self.bert_encoder=model
            lm_encoder_dim = model.config.hidden_size
        else:
            lm_encoder_dim = self.lm_feature_dim

        self.lm_encoder_dim = lm_encoder_dim

        ## set Criterion
        self.criterion = LOSS_MAPPING[label_cfg.loss_type]
        self.loss_type = label_cfg.loss_type

        ## Coeff
        self.coeff = model_cfg.coeff


    def w2v_encoding(self, audio, padding_mask=None, mask=None):
        if hasattr(self, 'w2v_encoder'):
            if self.wav2vec_name == 'wav2vec':
                input_args = {
                    "source": audio,
                    "padding_mask": padding_mask,
                    "mask": mask,
                }
                if self.finetune_wav2vec:
                    res = self.w2v_encoder.extract_features(**input_args)
                    audio, padding_mask = res['x'], res["padding_mask"]
                else:
                    with torch.no_grad():
                        self.w2v_encoder.eval()
                        res = self.w2v_encoder.extract_features(**input_args)
                        audio, padding_mask = res['x'], res["padding_mask"]
            if self.wav2vec_name == 'vq_wav2vec':
                if self.finetune_wav2vec:
                    audio = self.w2v_encoder.extract_features(audio)
                else:
                    with torch.no_grad():
                        self.w2v_encoder.eval()
                        audio = self.w2v_encoder.extract_features(audio)
        return audio, padding_mask


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
        language_feature_cfg = cfg.language_feature
        label_cfg = cfg.label

        model = cls(model_cfg, audio_feature_cfg, language_feature_cfg, label_cfg)
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

