import io
import os
from pathlib import Path
import librosa
from stt_api import RestAPI, APIConfig
from dataclasses import _MISSING_TYPE, dataclass, field
import logging
from google.cloud import speech

@dataclass
class GoogleConfig(APIConfig):
    name = 'google'

class GoogleAPI(RestAPI):
    def __init__(self, cfg:GoogleConfig):
        super(GoogleAPI, self).__init__(cfg)
        self.client = speech.SpeechClient()

    def _recognize(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None
        try:
            # normalize Audio
            #self._normalize_audio(audio_path, temp_path)

            temp_path = audio_path

            # Loads the audio into memory
            with io.open(temp_path, "rb") as audio_file:
                content = audio_file.read()
                audio = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.default_sample_rate,
                # 한국어
                #language_code="ko-KR",

                ### 영어로 변경
                language_code="en-US",
            )

            # Detects speech in the audio file
            response = self.client.recognize(config=config, audio=audio)

            response_texts = list()
            for result in response.results:
                response_texts.append(str(result.alternatives[0].transcript).strip())

            return " ".join(response_texts)
        except Exception as e:
            print("Problem in audio file [{}]".format(audio_path))
            print('[Failed message]' + str(e))
            return None

if __name__ == '__main__':
    print("시작하기")

    import argparse
    from omegaconf import OmegaConf
    import pandas as pd
    
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_path", default='kakao_multimodal_small.pkl', metavar="DIR",
            help="다운로드 한 파일의 위치"
        )
        parser.add_argument(
            "--save_path", default='kakao_multimodal_small.pkl', metavar="DIR",
            help="root directory containing flac files to index"
        )
        parser.add_argument(
            "--max_len", default=1000, type=int,
            help="root directory containing flac files to index"
        )
        parser.add_argument(
            "--user", default=None,
            help="select user"
        )
        return parser.parse_args()

    ## get argparse
    args = get_parser()

    cfg = OmegaConf.structured(GoogleConfig)

    ## 시간 설정
    cfg.max_len = args.max_len
    
    api = GoogleAPI(cfg)
    df = pd.read_pickle(args.data_path)
    df = api.run_stt(df)

    print("저장하기")
    df.to_pickle(args.save_path)
    print("저장 성공")

