import numpy as np
import soundfile as sf
from librosa import istft

from src.crnn_denoising.config import *  # CRNN's config used


class AudioRestorer:

    def revert_features_to_audio(self, features, phase, mean, std):
        # scale the outpus back to the original range
        if mean and std:
            features = std * features + mean
        phase = np.transpose(phase, (1, 0))

        features = np.squeeze(features)
        features = features * np.exp(1j * phase)
        features = np.transpose(features, (1, 0))

        return self._restore_audio(features)


    def _restore_audio(self, stft_features):
        return istft(stft_features, win_length=WINDOW_LENGTH, hop_length=OVERLAP, window='hamming', center=True)


    def write_audio(self, denoised, filename):
        path = f'audios_crnn/{filename}'
        sf.write(path, denoised, SAMPLE_RATE)
        return path
