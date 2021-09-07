import os
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment, effects
import soundfile as sf
import logging
from src.configs import BasicConfig
from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class APIConfig(BasicConfig):
    name:str = 'api'
    column_name:str='stt'
    audio_dir:str='data/voice_sentiment_4/audio'
    ext:str='wav'
    modified_flag:bool=True

class RestAPI:
    default_sample_rate=16000
    modified_ids = {
        '5e2979c25807b852d9e018d5': '5e37def1dbc4b7182a6a9e44',
        '5e298bc45807b852d9e01a10': '5e38a1d805fef317e874c5f7',
        '5e2998b85807b852d9e01b02': '5e37da3e33e9ad176cc9b2b1',
        '5e33638b5807b852d9e04aeb': '5e37e25905fef317e874c12e',
        '5e32924e5807b852d9e03894': '5e38d5c133e9ad176cc9b8e5',
        '5e2ad4145807b852d9e020d9': '5e3937ab05fef317e874c913',
        '5e31622f5807b852d9e032ba': '5e393ea333e9ad176cc9bae6',
        '5e2ad43e5807b852d9e020dc': '5e39383fdbc4b7182a6aa5f4',
        '5e298bdc5807b852d9e01a11': '5e38a2037995ef170fc0f96c',
        '5e298c085807b852d9e01a12': '5e38a219c8c25f16cd145d25',
        '5e3292825807b852d9e0389a': '5e38d655dbc4b7182a6aa405',
        '5e298b9f5807b852d9e01a0f': '5e38a184c8c25f16cd145d22',
        '5e315dca5807b852d9e03275': '5e380ef305fef317e874c3b5',
        '5e3292655807b852d9e03896': '5e38d5fe05fef317e874c705',
        '5e33a9d35807b852d9e050f4': '5e37e2f005fef317e874c13e',
        '5e3161c65807b852d9e032af': '5e393e55ee8206179943d383',
    }

    def __init__(self, cfg:APIConfig):
        self.name = cfg.name
        self.column_name = cfg.column_name
        self.audio_dir = cfg.audio_dir
        self.ext = cfg.ext
        self.modified_flag = cfg.modified_flag

        temp_folder = 'temp'
        os.makedirs(temp_folder, exist_ok=True)
        temp_path = os.path.join(temp_folder, "{}.wav".format(self.__class__.__name__))
        self.temp_path = temp_path


    def run_stt(self, script_df):
        logging.info('start to call Speech to Text API')

        ## for tqdm
        tqdm.pandas()

        if self.modified_flag:
            for wav_id, modified_id in self.modified_ids.items():
                mask = script_df['wav_id'] == wav_id
                script_df.loc[mask, 'wav_id'] = modified_id

        if 'stt' not in script_df.columns:
            script_df['stt'] = None
        ## find None rows
        null_rows = script_df['stt'].isnull()
        ## apply to stt
        if null_rows.sum() > 0:
            script_df.loc[null_rows, 'stt'] = script_df.loc[null_rows].progress_apply(
                lambda x: self._recognize(os.path.join(self.audio_dir, "{}.{}".format(x['wav_id'], self.ext)), self.temp_path), axis=1)

        logging.info('End Speech to Text API')
        return script_df

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