import os
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment, effects
import soundfile as sf
import logging
from src.configs import BasicConfig
from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class APIConfig:
    name:str = 'api'
    column_name:str='stt'
    ext:str='wav'
    modified_flag:bool=True
    max_len:int=10

class RestAPI:
    default_sample_rate=16000

    def __init__(self, cfg:APIConfig):
        self.name = cfg.name
        self.column_name = cfg.column_name
        self.ext = cfg.ext
        self.modified_flag = cfg.modified_flag

        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)
        temp_path = os.path.join(temp_folder, "{}.wav".format(self.__class__.__name__))
        self.temp_path = temp_path

        self.max_len=cfg.max_len


    def run_stt(self, script_df):
        logging.info('start to call Speech to Text API')
        print('start to call Speech to Text API')

        ## for tqdm
        tqdm.pandas()

        if 'stt' not in script_df.columns:
            script_df['stt'] = None
        ## find None rows
        null_rows = script_df['stt'].isnull()
        ## apply to stt
        if null_rows.sum() > 0:
            script_df.loc[null_rows, 'stt'] = script_df.loc[null_rows].progress_apply(
                lambda x: self._recognize(x['audio'], self.temp_path), axis=1)

        logging.info('End Speech to Text API')
        print('End Speech to Text API')
        return script_df

    def _normalize_audio(self, audio_path, temp_path):
        audio_extension = Path(audio_path).suffix.replace('.', '')
        sound_Obj = AudioSegment.from_file(audio_path, format=audio_extension)
        sound_Obj = sound_Obj.set_frame_rate(self.default_sample_rate)
        sound_Obj = sound_Obj.set_channels(1)
        sound_Obj = effects.normalize(sound_Obj)
        sound_Obj.export(temp_path, format="wav")

    def _id_to_wav(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            return None

        ## normalize Audio
        self._normalize_audio(audio_path, temp_path)
        wav, curr_sample_rate = sf.read(temp_path)

        return wav

    def _recognize(self, audio_path:str, temp_path:str):
        raise NotImplementedError

## normalize audio
# script_df['audio'] = None
# null_rows = script_df['audio'].isnull()
# if null_rows.sum() > 0:
#     script_df.loc[null_rows, 'audio'] = script_df.loc[null_rows].progress_apply(
#         lambda x: self._id_to_wav(os.path.join(self.audio_dir, "{}.{}".format(x['wav_id'], self.ext)),
#                             self.temp_path),
#         axis=1)