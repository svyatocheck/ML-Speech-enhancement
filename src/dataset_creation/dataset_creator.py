import os
import random
import warnings
from glob import glob

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.config import *


class Dataset_Creator:

    def __init__(self) -> None:
        self.sr = 16000
        self.snr = 50

    def execute_speech_preparation(self):
        speech_files = glob(f'{CLEAN_TRAIN}*.wav')
        noise_files = glob(f'{NOISE_TRAIN}*.wav')
        self._audio_preparation(speech_files, noise_files, RESULT_DATASET_TRAIN, 10)

        speech_files = glob(f'{CLEAN_TEST}*.wav')
        noise_files = glob(f'{NOISE_TEST}*.wav')
        self._audio_preparation(speech_files, noise_files, RESULT_DATASET_TEST, 5)
        
        
    def _audio_preparation(self, speech_files, noise_files, result_path, size):
        np.random.shuffle(noise_files)
        np.random.shuffle(speech_files)

        files_to_remove = list()    # remove bad audios

        # Change count of prepared audios
        for items in tqdm(speech_files[:size]):

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    audio = self._read_audio_file(items, self.sr)
                except RuntimeWarning:
                    # Remove audio if corrupted
                    files_to_remove.append(items)
                    continue
                except Exception:
                    continue

                try:
                    random_index = np.random.randint(0, len(noise_files))
                    noise = self._read_audio_file(noise_files[random_index], self.sr)
                except RuntimeWarning:
                    # Remove audio if corrupted
                    files_to_remove.append(noise)
                    continue
                except Exception:
                    continue

            # Make audio noisy, change SNR value here
            audio = self._mix_audio(audio, noise, random.randrange(0, self.snr, 10))

            fileName_absolute = os.path.splitext(os.path.basename(items))[0]
            filename_full = f'{result_path}{fileName_absolute}.wav'
            
            # Save noisy audio
            sf.write(filename_full, audio, self.sr)
            
            self._clear_corrupted_files(files_to_remove)
        print("Database created!")


    def _clear_corrupted_files(self, files):
        for files in tqdm(files):  # remove bad audios from dataset
            os.remove(files)


    def _mix_audio(self, speech, noise, snr):
        """
        Mixes the input speech and noise signals at a specified SNR.

        Args:
            speech: numpy array containing the speech signal
            noise: numpy array containing the noise signal
            snr: float representing the desired signal-to-noise ratio in decibels (dB)

        Returns:
            numpy array containing the mixed audio signal
        """
        noise = noise[np.arange(len(speech)) % len(noise)]
        noise = noise.astype(np.float32)
        speech = speech.astype(np.float32)
        signal_energy = np.mean(speech**2)
        noise_energy = np.mean(noise**2)
        g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
        a = np.sqrt(1 / (1 + g**2))
        b = np.sqrt(g**2 / (1 + g**2))
        return a * speech + b * noise


    def _read_audio_file(self, audio_file, sample_rate):
        '''Audio loading and normalization'''
        audio, _ = librosa.load(audio_file, sr=sample_rate)
        audio, _ = librosa.effects.trim(audio)
        audio = librosa.util.normalize(audio)
        return audio
