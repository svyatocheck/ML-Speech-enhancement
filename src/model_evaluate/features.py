import os

import librosa
import numpy as np
from tqdm import tqdm

from src.model_evaluate.constants import *


class FeatureExtractor:

    def start_preprocess(self, audio):
        """
        This function takes in an audio file and extracts features from it using other functions.
        :param audio: path to audio file
        :return: numpy array
        """
        spectrogram = self._make_spectrograms(audio)
        spectrogram = self._calculate_means(spectrogram)
        X = self._reshape_predictors(self._prepare_input_features(spectrogram))
        x_predictor = np.asarray(X).astype('float32')
        return x_predictor


    def _make_spectrograms(self, audio_path, normalize=True):
        """
        This function takes in an audio file and generates spectrograms using the librosa library.
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: spectrogram
        """
        audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        if normalize:
            audio_np = librosa.util.normalize(audio_np)
        return librosa.stft(y=audio_np, hop_length=OVERLAP, n_fft=N_FFT, center=True, window='hamming', win_length=WINDOW_LENGTH)


    def _calculate_means(self, spectrogram):
        '''
        Normalise stft spectrograms before feeding them to the DL model
        :param spectrogram: audio spectrogram in numpy array
        :return: encoded spectrogram
        Taken from article 1609.07132
        '''
        # noisy phase used to restore audio after DL model
        self.audio_phase = np.angle(spectrogram)

        # normilize stft spectrogram
        stft_feature = np.abs(spectrogram)
        mean = np.mean(stft_feature)
        std = np.std(stft_feature)

        # save them to restore previous values range after DL model
        self.mean = mean
        self.std = std
        return (stft_feature - mean) / std


    def _prepare_input_features(self, spectrogram):
        '''
        (Experimental) Phase aware scaling
        :param spectrogram: audio spectrogram in numpy array
        '''
        print(spectrogram.shape)
        stft_segments = np.zeros((N_FEATURES, N_SEGMENTS, spectrogram.shape[1] - N_SEGMENTS + 1))
        print(stft_segments.shape)
        
        for index in range(spectrogram.shape[1] - N_SEGMENTS + 1):
            stft_segments[:, :, index] = spectrogram[:,index:index + N_SEGMENTS]
        print(stft_segments.shape)
        return stft_segments


    def _reshape_predictors(self, items):
        '''
        Function to reshape features for NN
        :param items: numpy array - features
        :return: numpy array - prepared features [?, 257, 1, 1]
        '''
        predictors = np.reshape(items, (items.shape[0], items.shape[1], 1, items.shape[2]))
        predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
        print(predictors.shape)
        return predictors
