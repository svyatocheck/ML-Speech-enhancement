from glob import glob
import numpy as np
import random
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import warnings
from src.dataset_creation.config import *

'''Data preparation script. You need to configure paths, snr and other parameters on your own before launch it.'''


def mix_audio(speech, noise, snr):
    '''Наложение исходного аудио на шум.
    SNR: 0 - 40'''
    noise = noise[np.arange(len(speech)) % len(noise)]

    noise = noise.astype(np.float32)
    speech = speech.astype(np.float32)

    signal_energy = np.mean(speech**2)
    noise_energy = np.mean(noise**2)

    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))

    return a * speech + b * noise


def read_audio_file(audio_file, sample_rate):
    '''Загрузка и нормализация аудио'''
    audio, _ = librosa.load(audio_file, sr=sample_rate)
    audio, _ = librosa.effects.trim(audio)
    div_fac = 1 / np.max(np.abs(audio)) / 3.0
    audio = audio * div_fac
    return audio


def audio_preparation(speech_files, noise_files, result_path, size):
    np.random.shuffle(noise_files)
    np.random.shuffle(speech_files)

    files_to_remove = list()    # remove bad audios

    for items in tqdm(speech_files[:size]):  # Change count of prepared audios

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # there are problems with this audio, need to remove it
                audio = read_audio_file(items, SAMPLE_RATE)
            except RuntimeWarning:
                files_to_remove.append(items)
                continue

            random_index = np.random.randint(0, len(noise_files))
            try:
                noise = read_audio_file(noise_files[random_index], SAMPLE_RATE)
            except RuntimeWarning:
                files_to_remove.append(noise_files[random_index])
                continue

            # make audio noisy, change SNR value here
            audio = mix_audio(audio, noise, random.randrange(0, 30, 10))

            fileName_absolute = os.path.splitext(os.path.basename(items))[0]
            filename_full = f'{result_path}{fileName_absolute}.wav'

            sf.write(filename_full, audio, SAMPLE_RATE)  # save noisy audio

    for files in tqdm(files_to_remove):  # remove bad audios from dataset
        os.remove(files)


def execute_main():
    speech_files = glob(f'{SPEECH_DATASET}*.mp3') + glob(f'{CLEAN_TRAIN}*.wav')
    noise_files = glob(f'{NOISE_TRAIN}*/*.wav')
    audio_preparation(speech_files, noise_files, RESULT_DATASET_TRAIN, 20000)

    speech_files = glob(f'{CLEAN_TEST}*.wav')
    noise_files = glob(f'{NOISE_TEST}*.wav')
    audio_preparation(speech_files, noise_files, RESULT_DATASET_TEST, 500)


if __name__ == '__main__':
    execute_main()
