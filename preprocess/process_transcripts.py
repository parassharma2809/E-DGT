import pysrt
import os
import json
from tqdm import tqdm

TRANSCRIPT_FOLDER_PATH = '../data/transcript'
video_to_transcripts_dict = {}

for file in tqdm(os.listdir(TRANSCRIPT_FOLDER_PATH)):
    if file.endswith('.vtt'):
        subs = pysrt.open(f'{TRANSCRIPT_FOLDER_PATH}/{file}')
        sub_text = '\n'.join(
            [f'{sub.start.minutes}:{sub.start.seconds}-{sub.end.minutes}:{sub.end.seconds}: {sub.text}' for sub in
             subs])
        video_to_transcripts_dict[file.split('.')[0]] = sub_text

with open('video_to_transcripts.json', 'w') as f:
    json.dump(video_to_transcripts_dict, f, indent=3)
