import librosa
import numpy as np
import os
from tqdm import tqdm
from src.crnn_denoising.config import *


class FeatureInputGenerator:

    def start_preprocess(self, audio_files):
        '''Data preprocessing for the neural network.
        The input is a list of full names of already noisy audio recordings.
        The workflow is approximately as follows:
        1. Loading the audio recording (its noisy and original versions)
        2. Creating STFT based on the parameters specified in config.py
        3. Forming vectors of dimensionality (?, 257, 1, 1) for feeding to the NS.
        '''
        x_predictors = list()
        y_predictors = list()

        for audio in tqdm(audio_files):
            try:
                noisy_spectrogram = self._make_spectrograms(audio, False)
                
                # get clean file with the same as the noisy name.
                clean_file = self._load_clean(os.path.splitext(os.path.basename(audio))[0])
                clean_spectrogram = self._make_spectrograms(clean_file, True)
            except:
                continue

            noisy_spectrogram = self._calculate_means(noisy_spectrogram)
            clean_spectrogram = self._calculate_means(clean_spectrogram)

            X = self._prepare_input_features(noisy_spectrogram)
            Y = self._prepare_input_features(clean_spectrogram)

            X = self._reshape_features(X)
            Y = self._reshape_features(Y)

            x_predictors.append(X)
            y_predictors.append(Y)

        x_predictors = np.asarray(x_predictors, dtype=object)
        y_predictors = np.asarray(y_predictors, dtype=object)

        return np.concatenate(x_predictors), np.concatenate(y_predictors)


    def _read_audio_files(self, audio_path, normalize):
        """
        This function takes in an audio file and normalize it 
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: audio
        """
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        if normalize:
            audio, _ = librosa.effects.trim(audio)
            audio = librosa.util.normalize(audio)
        return audio


    def _make_spectrograms(self, audio_path, clean):
        """
        This function takes in an audio file and generates spectrograms using the librosa library.
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: spectrogram
        """
        audio_path = self._read_audio_files(audio_path, clean)
        stft = librosa.stft(y=audio_path, n_fft=N_FFT, win_length=WINDOW_LENGTH,
                            hop_length=OVERLAP, center=True, window='hamming')
        return stft


    def _calculate_means(self, spectrogram):
        '''
        Normalise stft spectrograms before feeding them to the DL model
        :param spectrogram: audio spectrogram in numpy array
        :return: encoded spectrogram
        '''
        stft_feature = np.abs(spectrogram)

        mean = np.mean(stft_feature)
        std = np.std(stft_feature)
        return (stft_feature - mean) / std


    def _prepare_input_features(self, spectrograms):
        '''
        Add new dimention to stft vectors
        :param spectrogram: audio spectrogram in numpy array
        '''
        stft_segments = np.zeros(
            (N_FEATURES, N_SEGMENTS, spectrograms.shape[1] - N_SEGMENTS + 1))

        for index in range(spectrograms.shape[1] - N_SEGMENTS + 1):
            stft_segments[:, :, index] = spectrograms[:,
                                                      index:index + N_SEGMENTS]
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
        abs_path_one = os.path.join(SPEECH_DATASET, f'{audio_name}.mp3')
        abs_path_two = os.path.join(CLEAN_TRAIN, f'{audio_name}.wav')
        abs_path_three = os.path.join(CLEAN_TEST, f'{audio_name}.wav')

        if os.path.exists(abs_path_one):
            return abs_path_one
        elif os.path.exists(abs_path_two):
            return abs_path_two
        elif os.path.exists(abs_path_three):
            return abs_path_three
        else:
            raise Exception(f'No clean file {audio_name} for the noisy file.')
