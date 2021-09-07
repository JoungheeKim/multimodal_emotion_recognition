from dataclasses import _MISSING_TYPE, dataclass, field


@dataclass
class AudioFeatureConfig:
    name:str='raw'

@dataclass
class MfccFeatureConfig(AudioFeatureConfig):
    name:str = 'mfcc'
    n_fft_size:int = 400
    n_mfcc:int = 80
    sample_rate:int = 16000

@dataclass
class Wav2vecFeatureConfig(AudioFeatureConfig):
    name: str = 'wav2vec'
    wav2vec_path:str = ''
    feature_dim=768



