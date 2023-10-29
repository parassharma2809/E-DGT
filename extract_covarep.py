from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def extract_covarep_features(audio_dir_path, output_dir_path):
    for audio_file in tqdm(os.listdir(audio_dir_path)):
        if audio_file.endswith('.wav'):
            # Read the audio file
            [Fs, x] = audioBasicIO.read_audio_file(f'{audio_dir_path}/{audio_file}')
            # Extract Covarep features
            features, feature_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
            # Save features to a CSV file
            df = pd.DataFrame(np.transpose(features), columns=feature_names)
            output_file = audio_file.split('.wav')[0]
            df.to_csv(f'{output_dir_path}/{output_file}.csv', index=False)


if __name__ == '__main__':
    extract_covarep_features('data/audio/wav', 'data/audio/extracted_features')
