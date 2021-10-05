## Insert Video_path
ROOT_PATH=video/

## Insert Save path
SAVE_PATH=data/multi_modal

for VIDEO_SUBDIR in 0001-0400 0401-0800 0801-1200 1201-1600 1601-2000 2001-2004 2401-2800 2801-3200 3201-3600 3601-4000 4001-4400 4401-4800 4801-5200 5201-5600; do
    python convert_video.py \
      --root_dir=${ROOT_PATH}/${VIDEO_SUBDIR} \
      --save_dir=${SAVE_PATH} \
      --save_frame_name=modified/raw/multimodal.pkl

done