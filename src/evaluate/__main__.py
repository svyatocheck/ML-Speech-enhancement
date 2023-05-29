import os
from glob import glob
from random import shuffle

import soundfile as sf
from pesq import pesq
import numpy as np
from pystoi import stoi
import tensorflow as tf
import librosa

from src.evaluate.config import *
from src.crnn_denoising.features import FeatureInputGenerator
from src.evaluate.restore_audio import AudioRestorer

noisy_audio_files = glob(NOISY_TEST + '*.wav')

shuffle(noisy_audio_files)

model = tf.keras.models.load_model('src/cnn_denoising/models/speech_model.h5')

generator = FeatureInputGenerator()
audio_restorer = AudioRestorer()

pesq_array = list()
stoi_array = list()

for audios in noisy_audio_files[:10]:
    x_audio = generator.start_preprocess(audios)
    results = model.predict(x_audio)    
    
    audio = audio_restorer.revert_features_to_audio(results, generator.audio_phase, generator.mean, generator.std)
    # audio_restorer.write_audio(audio, os.path.basename(audios))
    
    clean, _ = librosa.load(os.path.join(CLEAN_TEST, os.path.basename(audios)), sr=SAMPLE_RATE)
    div_fac = 1 / np.max(np.abs(clean)) / 3.0
    clean = clean * div_fac
    pesq_array.append(pesq(SAMPLE_RATE, clean[:audio.shape[0]], audio))
    stoi_array.append(stoi(clean[:audio.shape[0]], audio, SAMPLE_RATE))

print(f'STOI: {np.median(stoi_array)} PESQ: {np.median(pesq_array)}')