import io
import os
import sys
from glob import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import yaml
import torchaudio
import pandas as pd
import torch
from .base_dataset import BaseProcessor
import fairseq
from .base_dataset import FeatureConfig
from dataclasses import _MISSING_TYPE, dataclass, field
from pydub import AudioSegment, effects
import soundfile as sf
import logging
import numpy as np

@dataclass
class AudioFeatureConfig(FeatureConfig):
    name:str='raw'
    feature_dim:int = 1
    column_name:str = 'audio'
    audio_dir:str = '/code/gitRepo/sentiment_speech/data/sentiment_4/audio'
    ext:str='wav'
    default_sample_rate:int = 16000
    del_silence:bool = True
    min_len:float=0.0
    max_len:float=float(sys.maxsize)

@dataclass
class ExceptFeatureConfig(AudioFeatureConfig):
    name:str='except'
    feature_dim:int = 1

@dataclass
class MfccFeatureConfig(AudioFeatureConfig):
    name:str = 'mfcc'
    n_fft_size:int = 400
    n_mfcc:int = 80
    feature_dim = 80

@dataclass
class MelFeatureConfig(AudioFeatureConfig):
    name:str = 'mel'
    n_fft_size:int = 1024
    n_mels:int=80
    feature_dim:int = 80
    log_scale:bool = True

@dataclass
class Wav2vecFeatureConfig(AudioFeatureConfig):
    name: str = 'wav2vec'
    wav2vec_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/wav2vec_small.pt'
    feature_dim=768

@dataclass
class VQWav2vecFeatureConfig(AudioFeatureConfig):
    name: str = 'vq_wav2vec'
    wav2vec_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/english/vq-wav2vec_kmeans.pt'
    roberta_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/english/roberta'
    feature_dim=768

@dataclass
class VQTokenFeatureConfig(AudioFeatureConfig):
    name: str = 'vq_token'
    wav2vec_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/english/vq-wav2vec_kmeans.pt'
    roberta_path:str = '/code/gitRepo/sentiment_speech/pretrained_model/english/roberta'
    feature_dim=1


class AudioProcessor(BaseProcessor):
    def __init__(self, cfg:AudioFeatureConfig):
        super(AudioProcessor, self).__init__(cfg)
        self.config = cfg
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.feature_name = cfg.name
        self.column_name = cfg.column_name
        self.del_silence = cfg.del_silence

        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)
        temp_path = os.path.join(temp_folder, "{}.wav".format(self.__class__.__name__))
        self.temp_path = temp_path
        self.ext = cfg.ext
        self.audio_dir = cfg.audio_dir
        self.default_sample_rate = cfg.default_sample_rate
        self.min_len = cfg.min_len
        self.max_len=cfg.max_len

    def is_vq_wav2vec(self):
        if self.feature_name in ['vq_wav2vec', 'vq_token']:
            return True
        return False

    def _load_model(self):
        config = self.config
        if self.feature_name == 'raw':
            pass

        elif self.feature_name == 'except':
            pass

        elif self.feature_name == 'mfcc':
            self.device = torch.device('cpu')
            self.model = torchaudio.transforms.MFCC(sample_rate=config.default_sample_rate,
                                                        n_mfcc=config.n_mfcc,
                                                        log_mels=True,
                                                        melkwargs={'n_fft': config.n_fft_size}).to(self.device)
        elif self.feature_name == 'mel':
            self.device = torch.device('cpu')
            self.model = torchaudio.transforms.MelSpectrogram(sample_rate=config.default_sample_rate,
                                                        n_mels=config.n_mels,
                                                        n_fft=config.n_fft_size,
                                                        ).to(self.device)
            self.log_scale = config.log_scale

        elif config.name == 'wav2vec':
            assert os.path.isfile(config.wav2vec_path), 'wav2vec 2.0 파일이 없습니다. 다시한번 확인해 보세요. [{}]'.format(
                config.wav2vec_path)
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(config.wav2vec_path, None)
            w2v_args = state.get("cfg", None)
            task = fairseq.tasks.audio_pretraining.AudioPretrainingTask.setup_task(w2v_args.task)
            model = task.build_model(w2v_args.model)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            assert not model.training
            model.remove_pretraining_modules()
            model.to(self.device)
            self.model = model

        elif config.name in ['vq_wav2vec', 'vq_token']:
            state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(config.wav2vec_path, None)
            w2v_args = state.get("cfg", None)
            task = fairseq.tasks.audio_pretraining.AudioPretrainingTask.setup_task(w2v_args.task)
            model = task.build_model(w2v_args.model)
            model.load_state_dict(state["model"], strict=True)
            model.eval()

            model.to(self.device)
            self.model = model

            ## load roberta
            roberta = fairseq.models.roberta.RobertaModel.from_pretrained(config.roberta_path)
            roberta.eval()

            roberta.to(self.device)
            self.roberta = roberta

        else:
            raise ValueError('Need to add audio feature [{}]'.format(config.name))


    def _release_model(self):
        if hasattr(self, 'model'):
            self.model.to('cpu')
            del self.model

        if hasattr(self, 'roberta'):
            self.roberta.to('cpu')
            del self.roberta


    @torch.no_grad()
    def convert(self, audio_path, split_name=None):

        data = self._load_audio(audio_path, self.temp_path, split_name)
        if data is None:
            return None

        if self.feature_name == 'raw':
            return audio_path

        if self.feature_name == 'except':
            return np.array([0, 0])

        if self.feature_name == 'mfcc':
            source = torch.tensor(data, dtype=torch.float).to(self.device)
            vector = self.model(source)
            return vector.cpu().T.numpy()

        if self.feature_name == 'mel':
            source = torch.tensor(data, dtype=torch.float).to(self.device)
            vector = self.model(source)
            if self.log_scale:
                vector = torch.log(vector)
            return vector.cpu().T.numpy()

        if self.feature_name == 'wav2vec':
            source = torch.tensor(data, dtype=torch.float).to(self.device)
            source = source.unsqueeze(0)

            input_args = {
                "source": source,
                "padding_mask": None,
                "mask": None,
            }
            with torch.no_grad():
                res = self.model.extract_features(**input_args)
                x = res["x"]

            return x.squeeze(0).cpu().numpy()

        if self.feature_name == 'vq_wav2vec':
            source = torch.tensor(data, dtype=torch.float).to(self.device)
            source = source.unsqueeze(0)

            def indices_to_string(idxs):
                # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
                return "<s>" + " " + " ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))

            z = self.model.feature_extractor(source)
            _, idxs = self.model.vector_quantizer.forward_idx(z)

            idx_str = indices_to_string(idxs)
            tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True, add_if_not_exist=False).long()
            last_layer_features = self.roberta.extract_features(tokens)

            return last_layer_features.squeeze(0).cpu().numpy()

        if self.feature_name == 'vq_token':
            source = torch.tensor(data, dtype=torch.float).to(self.device)
            source = source.unsqueeze(0)

            def indices_to_string(idxs):
                # based on fairseq/examples/wav2vec/vq-wav2vec_featurize.py
                return "<s>" + " " + " ".join("-".join(map(str, a.tolist())) for a in idxs.squeeze(0))

            z = self.model.feature_extractor(source)
            _, idxs = self.model.vector_quantizer.forward_idx(z)

            idx_str = indices_to_string(idxs)
            tokens = self.roberta.task.source_dictionary.encode_line(idx_str, append_eos=True, add_if_not_exist=False)

            return tokens.cpu().numpy()

        return data

    def _normalize_audio(self, audio_path, temp_path):
        audio_extension = Path(audio_path).suffix.replace('.', '')
        sound_Obj = AudioSegment.from_file(audio_path, format=audio_extension)
        sound_Obj = sound_Obj.set_channels(1)
        sound_Obj = sound_Obj.set_frame_rate(self.default_sample_rate)
        #sound_Obj = effects.normalize(sound_Obj)
        sound_Obj.export(temp_path, format="wav")

    def _id_to_wav(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            return None

        ## normalize Audio
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

    def _load_audio(self, audio_path, temp_path, split_name=None):
        if not os.path.isfile(audio_path):
            return None

        try:
            audio_extension = Path(audio_path).suffix.replace('.', '')
            if audio_extension == 'raw':
                wav, curr_sample_rate = sf.read(audio_path, channels=1, samplerate=self.default_sample_rate, format='RAW', subtype='PCM_16')
            else:
                self._normalize_audio(audio_path, temp_path)
                wav, curr_sample_rate = sf.read(temp_path)
                
            if self.del_silence:
                non_silence_indices = librosa.effects.split(wav, top_db=30)
                wav = np.concatenate([wav[start:end] for start, end in non_silence_indices])
        except Exception as e:
            print(e)
            return None

        ## eliminate minlen
        if split_name == 'train':
            if len(wav) < (self.default_sample_rate * self.min_len):
                return None
            if len(wav) > (self.default_sample_rate * self.max_len):
                return None

        if self.feature_name == 'except':
            return np.array([0, 0])

        return wav

    def convert_data(self, script_df, split_name=None, **kwargs):
        ## load model
        self._load_model()

        ## for tqdm
        tqdm.pandas()

        #logging.info("load & normalize audio data")
        ##load data
        # script_df[self.column_name] = script_df['wav_id'].progress_apply(
        #      lambda x: self._id_to_wav(os.path.join(self.audio_dir, "{}.{}".format(x, self.ext)),
        #                              self.temp_path))

        # load data
        #logging.info("load & normalize audio data")
        #script_df[self.column_name] = script_df[self.column_name].progress_apply(
        #      lambda x: self._load_audio(x, self.temp_path, split_name))

        logging.info("preprocess audio feature")
        ## Convert data
        script_df[self.column_name] = script_df.progress_apply(
            lambda x: self.convert(x[self.column_name], split_name), axis=1)

        mask = ~script_df[self.column_name].isnull()
        script_df = script_df[mask]

        ## Release model
        self._release_model()
        return script_df

    def get_data(self, script_df):
        return script_df[self.column_name].values.tolist()
