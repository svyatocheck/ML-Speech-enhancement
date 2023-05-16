import librosa
import numpy as np

from src.evaluate.config import *


class FeatureGenerator:

    def start_preprocess(self, audio):
        try:
            spectrogram = self._make_spectrograms(audio)
        except Exception as ex:
            print(f"There is problem with file: {ex}")
            return

        self._put_phase(spectrogram)
        spectrogram = self._calculate_means(spectrogram)

        X = self._reshape_predictors(self._prepare_input_features(spectrogram))
        x_predictor = np.asarray(X).astype(np.float32)
        return x_predictor


    def _make_spectrograms(self, audio_path, normalize=False):
        """
        This function takes in an audio file and generates spectrograms using the librosa library.
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: spectrogram
        """
        audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        if normalize:
            div_fac = 1 / np.max(np.abs(audio_np)) / 3.0
            audio_np = audio_np * div_fac
        return librosa.stft(y=audio_np, hop_length=OVERLAP, n_fft=N_FFT, center=True, window='hamming')


    def _calculate_means(self, spectrogram):
        '''Preprocessing step for sound signal. 
        Taken from article 1609'''
        stft_feature = np.abs(spectrogram)
        mean = np.mean(stft_feature)
        std = np.std(stft_feature)
        self.mean = mean
        self.std = std
        return (stft_feature - mean) / std


    def _prepare_input_features(self, items):
        '''Формирование векторов из STFT диаграмм.'''
        stft_feature = np.concatenate(
            [items[:, 0:N_SEGMENTS-1], items], axis=1)
        stft_segments = np.zeros(
            (N_FEATURES, N_SEGMENTS, stft_feature.shape[1] - N_SEGMENTS + 1))
        for index in range(stft_feature.shape[1] - N_SEGMENTS + 1):
            stft_segments[:, :, index] = stft_feature[:,
                                                      index:index + N_SEGMENTS]
        return stft_segments


    def _reshape_predictors(self, items):
        '''Решейп векторов в требуемый НС формат.'''
        predictors = np.reshape(items, (items.shape[0], items.shape[1], 1, items.shape[2]))
        predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
        print(predictors.shape)
        return predictors


    def _put_phase(self, spectrogram):
        self.audio_phase = np.angle(spectrogram)
