
from .audio_dataset import AudioProcessor, AudioFeatureConfig
from .language_dataset import LanguageProcessor, LanguageFeatureConfig
from src.configs import DefaultConfig
from src.process.label_preprocess import LabelPreprocessConfig, LabelProcessor
from src.process.split_preprocess import SplitProcessor, SplitPreprocessConfig
import os

import torch
from src.utils import load_data
from torch.utils.data import (
    DataLoader, Dataset
)
import logging
import numpy as np
import sys
from typing import Tuple
import pickle
from pathlib import Path
from pydub import AudioSegment, effects
import soundfile as sf
import logging
import numpy as np
import librosa

def load_with_dump(base_cfg:DefaultConfig, split_name:str):
    dump_path = base_cfg.dump_path
    if not os.path.isdir(dump_path):
        dump_path = os.path.dirname(dump_path)
    file_path = os.path.join(dump_path, "{}.pkl".format(split_name))
    if os.path.isfile(file_path):
        logging.info('load dump file [{}]'.format(file_path))
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_with_dump(data:Tuple, base_cfg:DefaultConfig, split_name:str):
    dump_path = base_cfg.dump_path
    if not os.path.isdir(dump_path):
        dump_path = os.path.dirname(dump_path)

    if dump_path is None or dump_path == '':
        return

    os.makedirs(dump_path, exist_ok=True)
    file_path = os.path.join(dump_path, "{}.pkl".format(split_name))
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    logging.info('save dump file [{}]'.format(file_path))


class SentimentDataset(Dataset):
    def __init__(self, audio_data, language_data, add_language_data, label_data, max_audio_size=None, max_language_size=None, vq_wav2vec=False):
        self.audio_data = audio_data
        self.language_data = language_data
        self.add_language_data = add_language_data
        self.label_data = label_data

        self.max_audio_size = (
            max_audio_size if max_audio_size is not None else sys.maxsize
        )
        self.max_language_size = (
            max_language_size if max_language_size is not None else sys.maxsize
        )

        self.vq_wav2vec=vq_wav2vec
        self.pad_idx=0.0
        if vq_wav2vec:
            self.pad_idx = 1
            self.max_audio_size=2048


        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)
        self.temp_path = os.path.join(temp_folder, "{}.wav".format(self.__class__.__name__))

        self.del_silence = True
        self.default_sample_rate=16000

    def __len__(self):
        return len(self.label_data)

    def _normalize_audio(self, audio_path, temp_path):
        audio_extension = Path(audio_path).suffix.replace('.', '')
        sound_Obj = AudioSegment.from_file(audio_path, format=audio_extension)
        sound_Obj = sound_Obj.set_channels(1)
        sound_Obj = sound_Obj.set_frame_rate(self.default_sample_rate)
        #sound_Obj = effects.normalize(sound_Obj)
        sound_Obj.export(temp_path, format="wav")

    def _load_audio(self, audio_path):
        temp_path = self.temp_path
        audio_extension = Path(audio_path).suffix.replace('.', '')
        if audio_extension == 'raw':
            wav, curr_sample_rate = sf.read(audio_path, channels=1, samplerate=self.default_sample_rate, format='RAW', subtype='PCM_16')
        else:
            self._normalize_audio(audio_path, temp_path)
            wav, curr_sample_rate = sf.read(temp_path)
            
        if self.del_silence:
            non_silence_indices = librosa.effects.split(wav, top_db=30)
            wav = np.concatenate([wav[start:end] for start, end in non_silence_indices])
        
        return wav

    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        if type(audio) == str:
            audio = self._load_audio(audio)

        source = {
            'audio' : audio,
            'language': self.language_data[idx],
            'add_language' : self.add_language_data[idx] if self.add_language_data is not None else None,
            'label': self.label_data[idx],
        }
        return source

    @classmethod
    def load_with_config(cls, cfg, split_name='train'):
        split_names = ['train', 'valid', 'test']
        split_ids = {name:idx for idx, name in enumerate(split_names)}
        assert split_name in split_names, 'please insert valid split name [{}]'.format(split_names)

        base_cfg = cfg.base
        ad_cfg = cfg.audio_feature
        lm_cfg = cfg.language_feature
        label_cfg = cfg.label
        split_cfg = cfg.split

        tuple_data = load_with_dump(base_cfg, split_name)
        if tuple_data is not None:
            audio_data, language_data, add_language_data, label_data, vq_wav2vec = tuple_data
            return cls(audio_data, language_data, add_language_data, label_data, vq_wav2vec=vq_wav2vec)

        ## load dataset
        df = load_data(base_cfg)

        splitProcessor = SplitProcessor(split_cfg)
        split_dfs = splitProcessor.get_split_data(df)
        df = split_dfs[split_ids[split_name]]

        ## 이건 테스트 일단 추가용(실험)
        #print("train")
        #df = df[:int(len(df)/100)]
        #############################################


        ## label process
        labelProcessor = LabelProcessor(label_cfg)
        df = labelProcessor.convert_data(df)

        ## convert audio data
        audioProcessor = AudioProcessor(ad_cfg)
        df = audioProcessor.convert_data(df, split_name)

        ## convert language data
        languageProcessor = LanguageProcessor(lm_cfg)
        df = languageProcessor.convert_data(df, split_name)

        audio_data = audioProcessor.get_data(df)
        label_data = labelProcessor.get_data(df)
        language_data = languageProcessor.get_data(df, split_name)
        add_language_data = languageProcessor.get_add_data(df)

        logging.info("loaded dataset [{}]".format(split_name))

        vq_wav2vec=audioProcessor.is_vq_wav2vec()
        save_with_dump((audio_data, language_data, add_language_data, label_data, vq_wav2vec), base_cfg, split_name)

        return cls(audio_data, language_data, add_language_data, label_data, vq_wav2vec=vq_wav2vec)

    def collater(self, batch):
        ## audio
        audios = [b['audio'] for b in batch]
        channel = audios[0].shape[1] if len(audios[0].shape) >1 else False
        sizes = [len(s) for s in audios]

        target_size = min(max(sizes), self.max_audio_size)
        if channel:
            collated_audios = torch.zeros(len(audios), target_size, channel)
            padding_mask = (
                torch.BoolTensor(collated_audios.shape).fill_(False)
            )
            for i, (audio, size) in enumerate(zip(audios, sizes)):
                audio = torch.tensor(audio, dtype=torch.float)

                diff = size - target_size
                if diff == 0:
                    collated_audios[i] = audio
                elif diff < 0:
                    collated_audios[i] = torch.cat(
                        [audio, audio.new_full((-diff,channel), 0.0)]
                    )
                    padding_mask[i, diff:] = True
                else:
                    collated_audios[i] = self.crop_to_max_size(audio, target_size)
        else:
            collated_audios = torch.zeros(len(audios), target_size)
            padding_mask = (
                torch.BoolTensor(collated_audios.shape).fill_(False)
            )
            for i, (audio, size) in enumerate(zip(audios, sizes)):
                audio = torch.tensor(audio, dtype=torch.float)

                diff = size - target_size
                if diff == 0:
                    collated_audios[i] = audio
                elif diff < 0:
                    collated_audios[i] = torch.cat(
                        [audio, audio.new_full((-diff,), self.pad_idx)]
                    )
                    padding_mask[i, diff:] = True
                else:
                    collated_audios[i] = self.crop_to_max_size(audio, target_size)
            if self.vq_wav2vec:
                collated_audios=collated_audios.long()

        ## language
        tokens = [b['language'] for b in batch]
        input_ids = None
        token_type_ids = None
        attention_mask = None
        if type(tokens[0]) != str:
            channel = tokens[0].shape[1] if len(tokens[0].shape) > 1 else False
            sizes = [len(s) for s in tokens]
            target_size = min(max(sizes), self.max_language_size)
            if channel:
                input_ids = torch.zeros(len(tokens), target_size, channel)
                attention_mask = (
                    torch.BoolTensor(input_ids.shape[:2]).fill_(False)
                )
                for i, (token, size) in enumerate(zip(tokens, sizes)):
                    token = torch.tensor(token, dtype=torch.float)

                    diff = size - target_size
                    if diff == 0:
                        input_ids[i] = token
                    elif diff < 0:
                        input_ids[i] = torch.cat(
                            [token, token.new_full((-diff, channel), 0.0)]
                        )
                        attention_mask[i, diff:] = True
                    else:
                        input_ids[i] = self.crop_to_max_size(token, target_size)
            else:
                input_ids = torch.zeros(len(tokens), target_size).long()
                attention_mask = (
                    torch.LongTensor(input_ids.shape).fill_(1)
                )
                token_type_ids = (
                    torch.LongTensor(input_ids.shape).fill_(0)
                )
                for i, (token, size) in enumerate(zip(tokens, sizes)):
                    token = torch.tensor(token, dtype=torch.float)

                    diff = size - target_size
                    if diff == 0:
                        input_ids[i] = token
                    elif diff < 0:
                        input_ids[i] = torch.cat(
                            [token, token.new_full((-diff,), 0)]
                        )
                        attention_mask[i, diff:] = 0
                    else:
                        input_ids[i] = self.crop_to_max_size(token, target_size)

        ## add language
        add_tokens = [b['add_language'] for b in batch]
        add_input_ids = None
        add_token_type_ids = None
        add_attention_mask = None

        if type(add_tokens[0]) != str and add_tokens[0] is not None:
            channel = add_tokens[0].shape[1] if len(add_tokens[0].shape) > 1 else False
            sizes = [len(s) for s in add_tokens]
            target_size = min(max(sizes), self.max_language_size)
            if channel:
                add_input_ids = torch.zeros(len(add_tokens), target_size, channel)
                add_attention_mask = (
                    torch.BoolTensor(add_input_ids.shape[:2]).fill_(False)
                )
                for i, (add_token, size) in enumerate(zip(add_tokens, sizes)):
                    add_token = torch.tensor(add_token, dtype=torch.float)

                    diff = size - target_size
                    if diff == 0:
                        add_input_ids[i] = add_token
                    elif diff < 0:
                        add_input_ids[i] = torch.cat(
                            [add_token, add_token.new_full((-diff, channel), 0.0)]
                        )
                        add_attention_mask[i, diff:] = True
                    else:
                        add_input_ids[i] = self.crop_to_max_size(add_token, target_size)
            else:
                add_input_ids = torch.zeros(len(add_tokens), target_size).long()
                add_attention_mask = (
                    torch.LongTensor(add_input_ids.shape).fill_(1)
                )
                add_token_type_ids = (
                    torch.LongTensor(add_input_ids.shape).fill_(0)
                )
                for i, (add_token, size) in enumerate(zip(add_tokens, sizes)):
                    add_token = torch.tensor(add_token, dtype=torch.float)

                    diff = size - target_size
                    if diff == 0:
                        add_input_ids[i] = add_token
                    elif diff < 0:
                        add_input_ids[i] = torch.cat(
                            [add_token, add_token.new_full((-diff,), 0)]
                        )
                        add_attention_mask[i, diff:] = 0
                    else:
                        add_input_ids[i] = self.crop_to_max_size(add_token, target_size)

        labels = [b['label'] for b in batch]

        ## for BCE & Cross Entropy
        labels_dtype = torch.float if type(labels[0]) == list or type(labels[0])==np.ndarray else torch.long

        return {
                ## audio 속성
                'audio': collated_audios, 'padding_mask': padding_mask,

                ## language 속성
                'input_ids' : input_ids,
                'token_type_ids' : token_type_ids,
                'attention_mask' : attention_mask,

                ## add language 속성
                'add_input_ids': add_input_ids,
                'add_token_type_ids': add_token_type_ids,
                'add_attention_mask': add_attention_mask,

                ## label 속성
                'label': torch.tensor(labels, dtype=labels_dtype),
                }

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]


def load_dataloader(dataset:SentimentDataset, shuffle:bool=True, batch_size:int=2):

    ## 잠깐 수정
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collater
    )





