import os
from pathlib import Path
import requests
import json
from .stt_api import RestAPI, APIConfig
from dataclasses import _MISSING_TYPE, dataclass, field
import logging

@dataclass
class KaKaoConfig(APIConfig):
    name:str = 'kakao'
    kakao_key:str = '0dfec036628c8c40d61ead0bd108c469'

class KakaoAPI(RestAPI):
    def __init__(self, cfg:KaKaoConfig):
        super(KakaoAPI, self).__init__(cfg)
        self.kakao_key = cfg.kakao_key
        return

    def _recognize(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None
        try:
            # normalize Audio
            self._normalize_audio(audio_path, temp_path)

            # Loads the audio into memory
            kakao_speech_url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
            headers = {
                "Content-Type": "application/octet-stream",
                "X-DSS-Service": "DICTATION",
                "Authorization": "KakaoAK " + self.kakao_key,
            }
            with open(temp_path, 'rb') as fp:
                audio = fp.read()
            res = requests.post(kakao_speech_url, headers=headers, data=audio)

            result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}') + 1]
            result = json.loads(result_json_string)

            return result['value']

        except Exception as e:
            logging.info("Problem in audio file [{}]".format(audio_path))
            logging.info('[Failed message]' + str(e))
            return None

