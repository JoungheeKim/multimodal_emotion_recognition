from src.speech_to_text.clova_api import ClovaAPI
from src.speech_to_text.kakao_api import KakaoAPI
from src.speech_to_text.google_api import GoogleAPI
from .stt_api import APIConfig


api_name = {
    'clova' : ClovaAPI,
    'google' : GoogleAPI,
    'kakao' : KakaoAPI,
}

def get_api_class(cfg:APIConfig):
    name = cfg.name
    return api_name[name](cfg)