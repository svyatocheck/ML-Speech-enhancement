import os

import librosa
import numpy as np
from tqdm import tqdm

from src.config import *


class FeatureExtractor:

    def __init__(self, sr: float, wl: int, n_fft: int, hl: int, nf: int, ns, crnn : bool) -> None:
        """Extract feature before Machine Learning.
        Constants
        ----------
        sr (sample_rate): number > 0
            target sampling rate

        wl (window_length) : int <= n_fft [scalar]
            Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.

        n_fft : int > 0 [scalar]
            length of the windowed signal after padding with zeros. 

        hl (hop_length) : int > 0 [scalar]
            number of audio samples between adjacent STFT columns.

        nf (number_features) : int
            number of features in one input vector

        ns (number_segments) : int
            number of segments in one input vector.
        
        crnn : bool
            prepare data for crnn if true, for cnn if false."""
        self.sr = sr
        self.wl = wl
        self.n_fft = n_fft
        self.hl = hl
        self.nf = nf
        self.ns = ns
        self.crnn = crnn


    def start_preprocess(self, audio_files, save : bool):
        '''Data preprocessing for the neural network.
        The input is a list of full names of already noisy audio recordings.
        The workflow is approximately as follows:
        1. Loading the audio recording (its noisy and original versions)
        2. Creating STFT based on the parameters specified in config.py
        3. Forming vectors of dimensionality (?, ?, ?, 1) for feeding to the NN.
        '''
        x_preprocessed = list()
        y_preprocessed = list()

        for audio in tqdm(audio_files):
            # Load audio recording
            noisy_audio_data = self._read_audio_files(audio, False)
            clean_audio_data = self._read_audio_files(
                self._load_clean(os.path.splitext(
                    os.path.basename(audio))[0]), True
            )

            # Create STFT
            noisy_stft = librosa.stft(y=noisy_audio_data, n_fft=self.n_fft, win_length=self.wl,
                                      hop_length=self.hl, center=True, window='hamming')
            clean_stft = librosa.stft(y=clean_audio_data, n_fft=self.n_fft, win_length=self.wl,
                                      hop_length=self.hl, center=True, window='hamming')
            
            # Form vectors for NN input
            noisy_stft = self._calculate_means(noisy_stft)
            clean_stft = self._calculate_means(clean_stft)
            
            if self.crnn:
                X = self._prepare_input_features_crnn(noisy_stft)
                Y = self._prepare_input_features_crnn(clean_stft)
            else:
                X = self._prepare_input_features_cnn(noisy_stft)
                Y = self._prepare_input_features_cnn(clean_stft)
                
            X = self._reshape_features(X)
            Y = self._reshape_features(Y)

            x_preprocessed.append(X)
            y_preprocessed.append(Y)

        x_preprocessed = np.asarray(x_preprocessed, dtype=object)
        y_preprocessed = np.asarray(y_preprocessed, dtype=object)

        return np.concatenate(x_preprocessed).astype(np.float32), np.concatenate(y_preprocessed).astype(np.float32)


    def _read_audio_files(self, audio_path, normalize):
        """
        This function takes in an audio file and normalize it 
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: audio
        """
        audio, _ = librosa.load(audio_path, sr=self.sr)
        if normalize:
            audio, _ = librosa.effects.trim(audio)
            audio = librosa.util.normalize(audio)
        return audio


    def _calculate_means(self, stft_feature):
        '''
        Normalise stft spectrograms before feeding them to the DL model
        :param stft_feature: audio spectrogram, numpy array
        :return: encoded spectrogram
        '''
        stft_feature = np.abs(stft_feature)
        mean_values = np.mean(stft_feature)
        standard_deviation_values = np.std(stft_feature)
        return (stft_feature - mean_values) / standard_deviation_values


    def _prepare_input_features_crnn(self, spectrograms):
        '''
        Add new dimention to stft vectors
        :param spectrogram: audio spectrogram in numpy array
        '''
        stft_segments = np.zeros(
            (self.nf, self.ns, spectrograms.shape[1] - self.ns + 1)
        )

        for index in range(spectrograms.shape[1] - self.ns + 1):
            stft_segments[:, :, index] = spectrograms[:,index:index + self.ns]
        return stft_segments


    def _prepare_input_features_cnn(self, spectrograms):
        '''
        Feature extraction from STFT spectrograms.
        :param spectrogram: audio spectrogram in numpy array
        '''
        stft_feature = np.concatenate(
            [spectrograms[:, 0:self.ns-1], spectrograms], axis=1
        )
        
        stft_segments = np.zeros(
            (self.nf, self.ns, stft_feature.shape[1] - self.ns + 1)
        )
        
        for index in range(stft_feature.shape[1] - self.ns + 1):
            stft_segments[:, :, index] = stft_feature[:,index:index + self.ns]
        return stft_segments


    def _reshape_features(self, unprepared_features):
        '''
        Function to reshape features for NN
        :param items: features (numpy array)
        :return: prepared features [?, 255, 1, 1] in numpy array too
        '''
        features = np.reshape(unprepared_features, (
            unprepared_features.shape[0], unprepared_features.shape[1], 1, unprepared_features.shape[2]))
        return np.transpose(features, (3, 0, 1, 2)).astype(np.float32)


    def _load_clean(self, audio_name):
        """
        This function takes in an audio file name and returns it path in data directories
        :param name: path to audio file
        :return: audio_path
        """
        abs_path_two = os.path.join(CLEAN_TRAIN, f'{audio_name}.wav')
        abs_path_three = os.path.join(CLEAN_TEST, f'{audio_name}.wav')

        if os.path.exists(abs_path_two):
            return abs_path_two
        elif os.path.exists(abs_path_three):
            return abs_path_three
        else:
            raise Exception(f'No clean file {audio_name} for the noisy file.')
