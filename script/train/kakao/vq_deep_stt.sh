GPU_NUM=0
DATA_PATH=/code/gitRepo/sentiment_speech//modified/preprocess/kakao/multimodal_small/
LABEL_CLASS=multi6

for INDEX in 0 1 2 3 4;  do
    for ACCUMULATION_STEP in 16 32;  do
        python src/train.py \
            model=deep_fusion \
            model.wav2vec_name=vq_wav2vec \
            model.use_wav2vec=True \
            model.finetune_wav2vec=True \
            model.use_bert=True \
            model.finetune_bert=True \
            model.bert_path=/code/gitRepo/sentiment_speech//pretrained_model/HanBert-54kN-torch \
            model.roberta_path=/code/gitRepo/sentiment_speech//pretrained_model/english/roberta \
            model.hidden_dim=768 \
            model.am_concat_proj=conv \
            model.am_kernel_size=3 \
            model.lm_concat_proj=conv \
            model.lm_kernel_size=3 \
            model.am_last_hidden=False \
            model.lm_last_hidden=False \
            model.num_layers=1 \
            model.head_num=4 \
            audio_feature=vq_token \
            language_feature=hantoken \
            language_feature.column_name=stt \
            language_feature.bert_path=/code/gitRepo/sentiment_speech//pretrained_model/HanBert-54kN-torch \
            base.data_path=${DATA_PATH}/multimodal_small${INDEX}.pkl \
            base.save_path=/code/gitRepo/emotion_recognition/results/${GPU_NUM}/saved \
            base.experiments_path=/code/gitRepo/emotion_recognition/experiment/plan/kakao/${LABEL_CLASS}/kakao_vq_wav2vec_deepfusion_stt_${LABEL_CLASS}_english_768_4.csv \
            base.num_train_epochs=3 \
            base.learning_rate=5e-5 \
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