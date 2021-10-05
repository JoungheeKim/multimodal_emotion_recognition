import os
import json
from glob import glob
import pandas as pd
import subprocess
from pydub import AudioSegment, effects
import librosa
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from pathlib import Path
from moviepy.editor import AudioFileClip
from tqdm import tqdm
import argparse
import logging


def parse_data(json_path):
    results = list()
    with open(json_path, 'rb') as f:
        json_data = json.load(f)

    temp_data = json_data['data']

    temp_text = None
    temp_emotion = None
    start_index = None

    for index in range(1, int(json_data['nr_frame']) + 1):
        if str(index) not in temp_data:
            continue

        pass_flag = False
        for person_id, person_item in temp_data[str(index)].items():
            if 'emotion' in person_item and 'text' in person_item:
                pass_flag = True
                current_text = person_item['text']['script']
                if temp_text != current_text:
                    if start_index is None:
                        start_index = index
                        temp_text = current_text
                        temp_emotion = person_item['emotion']['multimodal']['emotion']
                        continue
                    result = {
                        'text': temp_text,
                        'start_idx': start_index,
                        'end_idx': index - 1,
                        'emotion': temp_emotion
                    }
                    start_index = index
                    temp_text = current_text
                    temp_emotion = person_item['emotion']['multimodal']['emotion']
                    results.append(result)
            if pass_flag:
                break
        if not pass_flag and start_index is not None:
            result = {
                'text': temp_text,
                'start_idx': start_index,
                'end_idx': index - 1,
                'emotion': temp_emotion,
            }
            results.append(result)

            start_index = None
            temp_text = None
            temp_emotion = None

    return results


def normlaize_audio(audio_path):
    sound_Obj = AudioSegment.from_file(audio_path, format='wav')
    sound_Obj = sound_Obj.set_channels(1)
    sound_Obj = sound_Obj.set_frame_rate(16000)
    #sound_Obj = effects.normalize(sound_Obj)
    sound_Obj.export(audio_path, format='wav')


def modfieid_videos(root_dir='raw', save_dir='modified', df=None):
    search_space = '{}/**/*.json'.format(root_dir)
    search_list = glob(search_space, recursive=True)

    if df is None:
        df = pd.DataFrame(columns=['audio', 'text', 'emotion'])

    for json_path in tqdm(search_list):

        video_path = json_path.replace('.json', '.mp4')
        save_path = json_path.replace(root_dir, save_dir)
        save_path = Path(save_path).parent
        os.makedirs(save_path, exist_ok=True)

        try:
            parsed_json = parse_data(json_path)
            clip = VideoFileClip(video_path)
            fps = clip.fps
            #duration = clip.duration
        except Exception as e:
            print("[ERROR 발생] {}".format(video_path))
            continue

        for idx, parse_info in enumerate(parsed_json):
            try:
                save_name = os.path.join(save_path, '{}.wav'.format(str(idx)))
                audio_start = int(parse_info['start_idx']) - 1
                audio_end = int(parse_info['end_idx']) - 1
                emotion = parse_info['emotion']
                text = parse_info['text']

                susbclip = clip.subclip(audio_start / fps, audio_end / fps)
                susbclip.audio.to_audiofile(save_name, verbose=False, logger=None)
                normlaize_audio(save_name)

                df.loc[-1] = [save_name, text, emotion]
                df.index = df.index + 1
            except Exception as e:
                print("[ERROR 발생] {}".format(video_path))

    return df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--save_frame_name",
        default='information.pkl',
        type=str,
        help="The input training data file (pickle).",
    )
    parser.add_argument(
        "--load_frame_name",
        default='information.pkl',
        type=str,
        help="The input training data file (pickle).",
    )
    return parser.parse_args()

def load_dataframe(temp_path:str):
    if temp_path is None or not os.path.isfile(temp_path):
        return None

    return pd.read_pickle(temp_path)

def save_dataframe(df:pd.DataFrame, temp_path):
    df.to_pickle(temp_path)

if __name__ == '__main__':
    args = get_parser()
    assert os.path.isdir(args.root_dir), '다시 확인해 보세요 [{}]'.format(args.root_dir)
    assert os.path.isdir(args.save_dir), '다시 확인해 보세요 [{}]'.format(args.save_dir)

    df = load_dataframe(args.load_frame_name)
    df = modfieid_videos(args.root_dir, args.save_dir, df)
    save_dataframe(df, args.save_frame_name)

    print("done")


