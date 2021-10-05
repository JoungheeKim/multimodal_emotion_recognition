GPU_NUM=0
DATA_PATH=/code/gitRepo/sentiment_speech//modified/preprocess/kakao/multimodal_small/
LABEL_CLASS=multi6
BATCH_SIZE=2
HEAD_NUM=16
NUM_LAYERS=8
LEARNING_RATE=5e-5
INDEX=0
KERNEL_SIZE=3

for HIDDEN_DIM in 80;  do
    for ACCUMULATION_STEP in 16 32 64;  do
        for KERNEL_SIZE in 3 5 7 9 11 ; do
            for HEAD_NUM in 16 10 8 5 ; do
                for NUM_LAYERS in 8 6 4 2; do
                    for LEARNING_RATE in 5e-2 5e-3 5e-4 5e-5; do
                        for INDEX in 0 1 2 3 4;  do
                            python src/train.py \
                                model=audio_transformer \
                                audio_feature=mfcc \
                                base.data_path=${DATA_PATH}/multimodal_small${INDEX}.pkl \
                                base.save_path=/code/gitRepo/emotion_recognition/results/kakao/audio_transformer_mfcc \
                                base.experiments_path=/code/gitRepo/emotion_recognition/experiment/kakao/${LABEL_CLASS}/mfcc/mfcc_${LABEL_CLASS}_acc${ACCUMULATION_STEP}_head${HEAD_NUM}_layer${NUM_LAYERS}_lr${LEARNING_RATE}_kernel${KERNEL_SIZE}_hidden${HIDDEN_DIM}.csv \
                                base.num_train_epochs=5 \
                                base.learning_rate=${LEARNING_RATE} \
                                base.train_batch_size=2 \
                                base.eval_batch_size=2 \
                                base.gradient_accumulation_steps=${ACCUMULATION_STEP} \
                                model.head_num=${HEAD_NUM} \
                                model.proj_layer=conv \
                                model.hidden_dim=80 \
                                model.kernel_size=${KERNEL_SIZE} \
                                model.num_layers=${NUM_LAYERS} \
                                model.use_wav2vec=False \
                                model.last_hidden=False \
                                base.logging_steps=200 \
                                label=${LABEL_CLASS} \
                                label.loss_type=cross \
                                audio_feature.min_len=0.5 \
                                audio_feature.max_len=8.0
                        done
                    done
                done
            done
        done
    done
done
