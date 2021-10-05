
DATA_PATH=modified/preprocess/kakao/multimodal_small.pkl
LABEL_CLASS=multi6
GPU_NUM=0

HEAD_NUM=2
HIDDEN_DIM=512

for ACCUMULATION_STEP in 16 32;  do
    for INDEX in 0 1 2 3 4;  do
        ## 한국어 SHALLOW
        python src/train.py \
            model=multimodal_transformer \
            model.wav2vec_name=wav2vec \
            model.use_wav2vec=True \
            model.finetune_wav2vec=True \
            model.wav2vec_path=pretrained_model/wav2vec_small.pt \
            model.use_bert=True \
            model.finetune_bert=True \
            model.bert_path=pretrained_model/HanBert-54kN-torch \
            model.num_layers=1 \
            model.head_num=${HEAD_NUM} \
            model.hidden_dim=${HIDDEN_DIM} \
            model.am_kernel_size=3 \
            model.am_concat_proj=conv \
            model.lm_kernel_size=3 \
            model.lm_concat_proj=conv \
            audio_feature=raw \
            language_feature=hantoken \
            language_feature.column_name=stt \
            language_feature.add_column_name=text \
            language_feature.bert_path=pretrained_model/HanBert-54kN-torch \
            base.data_path=${DATA_PATH} \
            base.save_path=results/${GPU_NUM}/saved \
            base.experiments_path=experiment/kakao/${LABEL_CLASS}/wav2vec_multitransformer_text_${LABEL_CLASS}_head${HEAD_NUM}_hidden${HIDDEN_DIM}.csv \
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