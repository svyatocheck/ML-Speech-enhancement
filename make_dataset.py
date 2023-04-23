from glob import glob
import numpy as np
import random
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import warnings
from constants import *

def _mix_audio(speech, noise, snr):
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


speech_files = glob(f'{TEST_DATASET}/*.wav')
noise_files = glob(f'{DATA_PATH}/input/noise/noise_test/*.wav')
result_files = f'{DATA_PATH}/noisy_test/'

np.random.shuffle(noise_files)

files_to_remove = list()
for items in tqdm(speech_files):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
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
        
        audio = _mix_audio(audio, noise, random.randrange(0, 50, 10))
        fileName_absolute = os.path.splitext(os.path.basename(items))[0]
        sf.write(f'{result_files}{fileName_absolute}.wav', audio, SAMPLE_RATE)


for files in tqdm(files_to_remove):
    os.remove(files)