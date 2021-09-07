import os
from tqdm import tqdm
import torchaudio
import torch
from .base_preprocess import BaseProcessor, BasicProcessConfig
from pathlib import Path
import librosa
from pydub import AudioSegment, effects
import soundfile as sf
from dataclasses import _MISSING_TYPE, dataclass, field
import numpy as np

@dataclass
class AudioPreprocess(BasicProcessConfig):
    name: str = 'audio_prerpocess'
    column_name: str = 'audio'
    audio_dir:str = '/code/gitRepo/sentiment_speech/data/sentiment_4/audio'
    ext:str='wav'
    default_sample_rate:int = 16000
    del_silence:bool = True




class AudioProcessor(BaseProcessor):
    def __init__(self, cfg:AudioPreprocess):
        super(AudioProcessor, self).__init__(cfg)
        self.default_sample_rate = cfg.default_sample_rate
        self.del_silence = cfg.del_silence
        self.ext = cfg.ext
        self.audio_dir=cfg.audio_dir
        self.column_name=cfg.column_name

        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)
        temp_path = os.path.join(temp_folder, "{}.wav".format(self.__class__.__name__))
        self.temp_path = temp_path


    def _normalize_audio(self, audio_path, temp_path):
        audio_extension = Path(audio_path).suffix.replace('.', '')
        sound_Obj = AudioSegment.from_file(audio_path, format=audio_extension)
        sound_Obj = sound_Obj.set_frame_rate(self.default_sample_rate)
        sound_Obj = effects.normalize(sound_Obj)
        sound_Obj.export(temp_path, format="wav")

    def _id_to_wav(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            return None

        ## normalize Audio
        self._normalize_audio(audio_path, temp_path)
        wav, curr_sample_rate = sf.read(temp_path)
        if self.del_silence:
            non_silence_indices = librosa.effects.split(wav, top_db=30)
            wav = np.concatenate([wav[start:end] for start, end in non_silence_indices])

        return wav

    def convert(self, data):
        vector = self.audio2vec(torch.Tensor(data, device=self.device))
        return vector.cpu().T.numpy()

    def convert_data(self, script_df):
        ## for tqdm
        tqdm.pandas()

        script_df['audio'] = script_df.progress_apply(
            lambda x: self.convert(x['audio']), axis=1)
        print('done')

        return script_df






