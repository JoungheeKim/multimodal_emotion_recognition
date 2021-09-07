import io
import os
from pathlib import Path
import librosa
from .stt_api import RestAPI, APIConfig
from dataclasses import _MISSING_TYPE, dataclass, field
import logging

@dataclass
class GoogleConfig(APIConfig):
    name = 'google'

class GoogleAPI(RestAPI):
    def __init__(self, cfg:GoogleConfig):
        super(GoogleAPI, self).__init__(cfg)
        from google.cloud import speech
        self.client = speech.SpeechClient()

    def _recognize(self, audio_path, temp_path):
        if not os.path.isfile(audio_path):
            print("NO audio file [{}]".format(audio_path))
            return None
        try:
            # normalize Audio
            self._normalize_audio(audio_path, temp_path)

            # Loads the audio into memory
            with io.open(temp_path, "rb") as audio_file:
                content = audio_file.read()
                audio = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.default_sample_rate,
                language_code="ko-KR",
            )

            # Detects speech in the audio file
            response = self.client.recognize(config=config, audio=audio)

            response_texts = list()
            for result in response.results:
                response_texts.append(str(result.alternatives[0].transcript).strip())

            return " ".join(response_texts)
        except Exception as e:
            logging.info("Problem in audio file [{}]".format(audio_path))
            logging.info('[Failed message]' + str(e))
            return None
