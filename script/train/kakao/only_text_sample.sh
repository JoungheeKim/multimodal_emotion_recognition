GPU_NUM=0
DATA_PATH=/code/gitRepo/sentiment_speech//modified/preprocess/kakao/multimodal_small/
LABEL_CLASS=multi6
BATCH_SIZE=2
HEAD_NUM=10
NUM_LAYERS=8
LEARNING_RATE=5e-5
INDEX=0


## 한국어 SHALLOW
python src/train.py \
    model=lm_transformer \
    model.use_bert=False \
    audio_feature=except \
    language_feature=fasttext \
    language_feature.column_name=text \
    base.data_path=${DATA_PATH}/multimodal_small${INDEX}.pkl \
    base.save_path=/code/gitRepo/emotion_recognition/results/google/raw_token_transformer_lm \
    base.experiments_path=/code/gitRepo/emotion_recognition/experiment/kakao/${LABEL_CLASS}/fasttext_text_${LABEL_CLASS}_batch${BATCH_SIZE}_head${HEAD_NUM}_layer${NUM_LAYERS}_lr${LEARNING_RATE}.csv \
    base.num_train_epochs=5 \
    base.learning_rate=${LEARNING_RATE} \
    base.train_batch_size=${BATCH_SIZE} \
    base.eval_batch_size=${BATCH_SIZE} \
    model.num_layers=${NUM_LAYERS} \
    model.head_num=${HEAD_NUM} \
    model.last_hidden=False \
    base.logging_steps=200 \
    label=${LABEL_CLASS} \
    label.loss_type=cross