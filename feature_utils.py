import numpy as np
import librosa

def extract_features(audio):
    sr = 16000

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    flatness_mean = np.mean(spectral_flatness)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    features = np.concatenate([mfcc_mean, [flatness_mean], [zcr_mean]])
    return features.reshape(1, -1)
