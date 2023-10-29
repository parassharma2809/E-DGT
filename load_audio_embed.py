import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle as pkl


pth = 'data/audio/extracted_features'
audio_feat_dict = {}
for p in tqdm(os.listdir(pth)):
    if p.endswith('.csv'):
        df = pd.read_csv(f'{pth}/{p}')
        ra = [row.values for _, row in df.iterrows()]
        ra = np.array(ra)
        audio_feat_dict[p.split('.csv')[0]] = ra

with open('preprocess/audio_feat_dict.pkl', 'wb') as f:
    pkl.dump(audio_feat_dict, f)
