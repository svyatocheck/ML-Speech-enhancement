from cnn_denoising.config import *
import librosa
import numpy as np
import os
from tqdm import tqdm


class FeatureInputGenerator:

    def start_preprocess(self, audio_files):
        '''Предобработка данных для нейронной сети.
        На вход подается список полных имен уже зашумленных аудиозаписей.
        Ход работ примерно следующий:
        1. Загрузка аудиозаписи (ее зашумленная и исходная версии)
        2. Создание STFT на основе параметров заданных в constants.py
        3. Формирование векторов размерности (?, 129, 8, 1) для подачи в НС.'''
        x_predictors = list()
        y_predictors = list()

        for audio in tqdm(audio_files):
            try:
                noisy_spectrogram = self._make_spectrograms(audio)
                # get clean file with the same as the noisy name.
                clean_file = self._load_clean(
                    os.path.splitext(os.path.basename(audio))[0])
                clean_spectrogram = self._make_spectrograms(clean_file, True)
            except:
                continue

            noisy_spectrogram = self._calculate_means(noisy_spectrogram)
            clean_spectrogram = self._calculate_means(clean_spectrogram)

            X = self._prepare_input_features(noisy_spectrogram)
            Y = self._prepare_input_features(clean_spectrogram)

            X = self._reshape_predictors(X)
            Y = self._reshape_predictors(Y)

            x_predictors.append(X)
            y_predictors.append(Y)

        x_predictors = np.asarray(x_predictors, dtype=object)
        y_predictors = np.asarray(y_predictors, dtype=object)

        return np.concatenate(x_predictors).astype(np.float32), np.concatenate(y_predictors).astype(np.float32)

    def _make_spectrograms(self, audio, clean=False):
        '''Создание STFT диаграмм.'''
        audio = self._read_audio_files(audio, clean)
        stft = librosa.stft(y=audio, n_fft=N_FFT,
                            hop_length=OVERLAP, center=True, window='hamming')
        return stft

    def _read_audio_files(self, path, normalize):
        '''Загрузка, удаление тихих участков из аудио файла, нормализация.'''
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        if normalize:
            audio, _ = librosa.effects.trim(audio)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
        return audio

    def _load_clean(self, name):
        '''Поиск исходной аудиозаписи по заданным директориям.'''
        abs_path_one = os.path.join(SPEECH_DATASET, f'{name}.mp3')
        abs_path_two = os.path.join(CLEAN_TRAIN, f'{name}.wav')
        abs_path_three = os.path.join(CLEAN_TEST, f'{name}.wav')

        if os.path.exists(abs_path_one):
            return abs_path_one
        elif os.path.exists(abs_path_two):
            return abs_path_two
        elif os.path.exists(abs_path_three):
            return abs_path_three
        else:
            raise Exception(f'No clean file {name} for the noisy file.')

    def _calculate_means(self, spectrogram):
        '''Необходимый в предобработке звукового сигнала этап.
        Взят из статьи 1609.'''
        stft_feature = np.abs(spectrogram)
        mean = np.mean(stft_feature)
        std = np.std(stft_feature)
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
        predictors = np.reshape(
            items, (items.shape[0], items.shape[1], 1, items.shape[2]))
        predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
        # print(predictors.shape)
        return predictors
