GPU_NUM=0
DATA_PATH=/code/gitRepo/sentiment_speech//modified/preprocess/kakao/multimodal_small/
LABEL_CLASS=multi6
KERNEL_SIZE=3
HIDDEN_DIM=80


for ACCUMULATION_STEP in 16 32;  do
    for LEARNING_RATE in 5e-4 5e-5; do
        for INDEX in 0 1 2 3 4;  do
            python src/train.py \
                model=shallow_fusion \
                audio_feature=mfcc \
                language_feature=fasttext \
                language_feature.column_name=text \
                base.data_path=${DATA_PATH}/multimodal_small${INDEX}.pkl \
                base.save_path=/code/gitRepo/emotion_recognition/results/kakao/shallow_fusion_mfcc_fasttext\
                base.experiments_path=/code/gitRepo/emotion_recognition/experiment/kakao/${LABEL_CLASS}/mfcc_fasttext/shallow_mfcc_fasttext${LABEL_CLASS}_acc${ACCUMULATION_STEP}_lr${LEARNING_RATE}_kernel${KERNEL_SIZE}_hidden${HIDDEN_DIM}.csv \
                base.num_train_epochs=3 \
                base.learning_rate=${LEARNING_RATE} \
                base.train_batch_size=2 \
                base.eval_batch_size=2 \
                base.gradient_accumulation_steps=${ACCUMULATION_STEP} \
                model.am_concat_proj=conv \
                model.am_kernel_size=${KERNEL_SIZE} \
                model.lm_concat_proj=conv \
                model.lm_kernel_size=${KERNEL_SIZE} \
                model.am_last_hidden=False \
                model.lm_last_hidden=False \
                model.hidden_dim=${HIDDEN_DIM} \
                model.use_wav2vec=False \
                model.use_bert=False \
                base.logging_steps=200 \
                label=${LABEL_CLASS} \
                label.loss_type=cross \
                audio_feature.min_len=0.5 \
                audio_feature.max_len=8.0
        done
    done
done

LAYERS=2
HEAD_NUM=10
for ACCUMULATION_STEP in 16 32;  do
    for LEARNING_RATE in 5e-4 5e-5; do
        for INDEX in 0 1 2 3 4;  do      
            python src/train.py \
                model=multimodal_transformer \
                audio_feature=mfcc \
                language_feature=fasttext \
                language_feature.column_name=text \
                model.use_wav2vec=False \
                model.finetune_wav2vec=False \
                model.use_bert=False \
                model.finetune_bert=False \
                model.num_layers=${LAYERS} \
                model.head_num=${HEAD_NUM} \
                model.hidden_dim=${HIDDEN_DIM} \
                model.am_kernel_size=${KERNEL_SIZE} \
                model.am_concat_proj=conv \
                model.lm_kernel_size=${KERNEL_SIZE} \
                model.lm_concat_proj=conv \
                base.data_path=${DATA_PATH}/multimodal_small${INDEX}.pkl \
                base.save_path=/code/gitRepo/emotion_recognition/results/kakao/multimodal_transformer_fasttext\
                base.experiments_path=/code/gitRepo/emotion_recognition/experiment/kakao/${LABEL_CLASS}/mfcc_fasttext/multi_mfcc_fasttext${LABEL_CLASS}_acc${ACCUMULATION_STEP}_lr${LEARNING_RATE}_kernel${KERNEL_SIZE}_hidden${HIDDEN_DIM}_head${HEAD_NUM}_layer${LAYERS}.csv \
                base.num_train_epochs=3 \
                base.learning_rate=${LEARNING_RATE} \
                base.gradient_accumulation_steps=${ACCUMULATION_STEP} \
                base.train_batch_size=2 \
                base.eval_batch_size=2 \
                base.logging_steps=200 \
                label=${LABEL_CLASS} \
                label.loss_type=cross \
                audio_feature.min_len=0.5 \
                audio_feature.max_len=8.0
        done
    done
done

