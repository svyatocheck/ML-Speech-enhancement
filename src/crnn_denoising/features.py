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
            
            X = self._reshape_predictors(X)
            Y = self._reshape_predictors(Y)
            
            x_predictors.append(X)
            y_predictors.append(Y)
        
        x_predictors = np.asarray(x_predictors, dtype=object)
        y_predictors = np.asarray(y_predictors, dtype=object)
        
        return np.concatenate(x_predictors), np.concatenate(y_predictors)
        
    
    def _read_audio_files(self, path, normalize):
        """
        This function takes in an audio file and normalize it 
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: audio
        """
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        if normalize:
            audio, _ = librosa.effects.trim(audio)
            div_fac = 1 / np.max(np.abs(audio)) / 3.0
            audio = audio * div_fac
        return audio
    
    
    def _load_clean(self, name):
        """
        This function takes in an audio file name and returns it path in data directories
        :param name: path to audio file
        :return: audio_path
        """
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
    
    
    def _make_spectrograms(self, audio, clean):
        """
        This function takes in an audio file and generates spectrograms using the librosa library.
        :param audio_path: path to audio file
        :param normalize: whether or not to normalize the spectrograms
        :return: spectrogram
        """
        audio = self._read_audio_files(audio, clean)
        stft = librosa.stft(y=audio, n_fft=N_FFT, win_length=WINDOW_LENGTH, hop_length=OVERLAP, center=True, window='hamming')
        return stft
    
    
    def _calculate_means(self, spectrogram):
        '''
        Important step to avoid extreme differences (more than 45 degree) between the noisy and clean phase
        :param spectrogram: audio spectrogram in numpy array
        :return: encoded spectrogram
        Taken from article 1609.07132
        '''
        stft_feature = np.abs(spectrogram)
        mean = np.mean(stft_feature)
        std = np.std(stft_feature)
        return (stft_feature - mean) / std

    
    def _prepare_input_features(self, spectrograms):
        '''
        Feature extraction from STFT spectrograms.
        :param spectrogram: audio spectrogram in numpy array
        '''
        stft_feature = np.concatenate([spectrograms[:,0:N_SEGMENTS-1], spectrograms], axis=1)
        stft_segments = np.zeros((N_FEATURES, N_SEGMENTS , stft_feature.shape[1] - N_SEGMENTS + 1))
        for index in range(stft_feature.shape[1] - N_SEGMENTS + 1):
            stft_segments[:,:,index] = stft_feature[:,index:index + N_SEGMENTS]
        return stft_segments
    
    
    def _reshape_predictors(self, items):
        '''
        Function to reshape features for NN
        :param items: numpy array - features
        :return: numpy array - prepared features [?, 255, 1, 1]
        '''
        predictors = np.reshape(items, (items.shape[0], items.shape[1], 1, items.shape[2]))
        predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
        return predictors
        