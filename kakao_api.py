import os
from pathlib import Path
import requests
import json
from stt_api import RestAPI, APIConfig
from dataclasses import _MISSING_TYPE, dataclass, field
import logging
import librosa
import soundfile as sf

@dataclass
class KaKaoConfig(APIConfig):
    name:str = 'kakao'
    kakao_key:str = ''



class KakaoAPI(RestAPI):
    def __init__(self, cfg:KaKaoConfig):
        super(KakaoAPI, self).__init__(cfg)
        self.kakao_key = cfg.kakao_key
        print(self.kakao_key)
        self.exceed = False
        self.max_len = cfg.max_len
        return
        
    def _split_recognize(self, audio_path, temp_path):
        if self.exceed:
            return None

        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None

        try:
            wav, samplerate = librosa.load(audio_path)
            split_points = librosa.effects.split(wav, top_db=20)

            results = []

            for split_point in split_points:
                start_point = split_point[0]
                end_point = split_point[1]

                temp_wav = wav[start_point:end_point]
                sf.write(temp_path, temp_wav, samplerate)
                temp_results = self._recognize(temp_path, temp_path)

                if temp_results is not None and temp_results != ' ':
                    results.append(temp_results)

            return ' '.join(results)

        except Exception as e:
            print('_split_recognize ERROR', str(e))
            logging.info("Problem in audio file [{}]".format(audio_path))
            logging.info('[Failed message]' + str(e))
            return None


    def _recognize(self, audio_path, temp_path):
        if self.exceed:
            return None

        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None
        try:
            # normalize Audio
            self._normalize_audio(audio_path, temp_path)

            ## 오디오 길이 확인
            duration = librosa.get_duration(filename=temp_path)
            if duration > self.max_len:
                print('초과')
                return None

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
            if res.ok:
                result_hint = '{"type":"finalResult"'
                if result_hint not in res.text:
                    return ' '

                result_json_string = res.text[res.text.index(result_hint):res.text.rindex('}') + 1]
                result = json.loads(result_json_string)
                return result['value']
            else:

                if res.status_code == 413:
                    ## 잘라서 넣기.
                    return self._split_recognize(audio_path, temp_path)

                if int(eval(res.text)['code']) == -10:
                    self.exceed = True
                    print("벌써 초과했다고?")
                elif int(eval(res.text)['code']) == -3:
                    self.exceed = True
                    print("서비스 신청을 안했네")

            return None

        except Exception as e:
            print("_recognize ERROR", str(e))
            logging.info("Problem in audio file [{}]".format(audio_path))
            logging.info('[Failed message]' + str(e))
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

    ## 시간 설정
    cfg.max_len = args.max_len
    
    api = KakaoAPI(cfg)
    df = pd.read_pickle(args.data_path)
    df = api.run_stt(df)

    print("저장하기")
    df.to_pickle(args.save_path)
    print("저장 성공")

