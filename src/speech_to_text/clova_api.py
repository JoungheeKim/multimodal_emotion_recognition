import os
from pathlib import Path
import requests
import json
from .stt_api import RestAPI, APIConfig
from dataclasses import _MISSING_TYPE, dataclass, field
import logging

@dataclass
class ClovaConfig(APIConfig):
    name = 'clova'

    ##무료
    #invoke_url:str = 'https://clovaspeech-gw.ncloud.com/external/v1/502/d9ba4040c758361d09fa3b1fecf0d3e60eb14bbff771d5aff994a574ad7f9dcb'
    #secret:str = '0c777760c4a84ceaadb7bb36f9cd4e93'

    ##유료
    invoke_url: str = 'https://clovaspeech-gw.ncloud.com/external/v1/511/d523db5e8fe5908d92b93235fec5f8859d81daa32eb335249fe514f0dd7e654c'
    secret: str = '58569c2786e14c72866963088cc6e8a6'


class ClovaAPI(RestAPI):
    def __init__(self, cfg:ClovaConfig):
        super(ClovaAPI, self).__init__(cfg)
        self.invoke_url = cfg.invoke_url
        self.secret = cfg.secret

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, sttEnable=True,
                wordAlignment=True, fullText=True, script='', diarization=None, keywordExtraction=None,
                groupByAudio=False):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'sttEnable': sttEnable,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'script': script,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'keywordExtraction': keywordExtraction,
            'groupByAudio': groupByAudio,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           sttEnable=True,
                           wordAlignment=True, fullText=True, script='', diarization=None, keywordExtraction=None,
                           groupByAudio=False):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'sttEnable': sttEnable,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'script': script,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'keywordExtraction': keywordExtraction,
            'groupByAudio': groupByAudio,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   sttEnable=True,
                   wordAlignment=True, fullText=True, script='', diarization=None, keywordExtraction=None,
                   groupByAudio=False):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'sttEnable': sttEnable,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'script': script,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'keywordExtraction': keywordExtraction,
            'groupByAudio': groupByAudio,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response

    def _recognize(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None
        try:
            # normalize Audio
            self._normalize_audio(audio_path, temp_path)

            ## call api
            res = self.req_upload(file=temp_path, completion='sync')
            results = json.loads(res.text)
            if 'text' not in results:
                logging.info("[FAILED] {}".format(audio_path))
                logging.info(results['message'])
                return None

            return results['text']

        except Exception as e:
            logging.info("Problem in audio file [{}]".format(audio_path))
            logging.info('[Failed message]' + str(e))
            return None


