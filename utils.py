import numpy as np
import librosa

def extract_mfcc_from_path(filepath):
    y, sr = librosa.load(filepath, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(40, 1)