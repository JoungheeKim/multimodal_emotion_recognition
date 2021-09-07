from .audio_model import AudioModelConfig
from .wav2keyword import Wav2Keyword
from .conformer import Conformer
from .lm_transformer import LMTransformerModel
from .audio_transformer import AudioTransformerModel
from .ad_lm_transformer import AudioLanguageTransformerModel
from .audio_lm_shallow import AudioLanguageShallowModel
from .audio_lm_shallow_point import AudioLanguageShallowPointModel
from .audio_lm_shallow_multi import AudioLanguageShallowMultiModel
from .audio_lm_shallow_multi_ver2 import AudioLanguageShallowMultiV2Model
from .audio_lm_shallow_pointer import AudioLanguageShallowPointerModel
from .ad_lm_transformer_pointer import AudioLanguageTransformerPointerModel
from .audio_lm_deep import AudioLanguageDeepModel
from .audio_lm_deep_pointer import AudioLanguageDeepPointerModel
from .audio_lm_deep_pointer_ver2 import AudioLanguageDeepPointerV2Model

## In here you must sign your model with name
model_name = {
    'wav2keyword' : Wav2Keyword,
    'conformer' : Conformer,
    'lm_transformer' : LMTransformerModel,
    'audio_transformer' : AudioTransformerModel,
    'multimodal_transformer' : AudioLanguageTransformerModel,
    'multimodal_transformer_pointer' : AudioLanguageTransformerPointerModel,
    'shallow_fusion' : AudioLanguageShallowModel,
    'shallow_fusion_point' : AudioLanguageShallowPointModel,
    'shallow_fusion_pointer' : AudioLanguageShallowPointerModel,
    'shallow_fusion_multi' : AudioLanguageShallowMultiModel,
    'shallow_fusion_multi_ver2' : AudioLanguageShallowMultiV2Model,
    'deep_fusion' : AudioLanguageDeepModel,
    'deep_fusion_pointer' : AudioLanguageDeepPointerModel,
    'deep_fusion_pointer_ver2' : AudioLanguageDeepPointerV2Model,
}

def get_model_class(cfg:AudioModelConfig):
    name = cfg.name
    return model_name[name]