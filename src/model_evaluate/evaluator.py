import os
from glob import glob
from random import shuffle

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from pesq import pesq
from pystoi import stoi

from src.config import *
from src.crnn_denoising.config import *
from src.model_evaluate.features import FeatureExtractor
from src.model_evaluate.restore_audio import AudioRestorer


class ModelEvaluator:

    def __init__(self, noisy_files, model_path="src/crnn_denoising/models/speech_model.h5") -> None:
        self.model = model_path
        self.noisy_files = noisy_files
        self.feature_extractor = FeatureExtractor()

    def evaluate(self):
        model = tf.keras.models.load_model(self.model)
        audio_restorer = AudioRestorer()

        pesq_array = list()
        stoi_array = list()

        for audios in self.noisy_files[:5]:
            x_audio = self.feature_extractor.start_preprocess(audios)
            results = model.predict(x_audio)

            audio = audio_restorer.revert_features_to_audio(
                results, self.feature_extractor.audio_phase, self.feature_extractor.mean, self.feature_extractor.std)
            # audio_restorer.write_audio(audio, os.path.basename(audios))

            clean, _ = librosa.load(os.path.join(CLEAN_TRAIN, os.path.basename(audios)), sr=SAMPLE_RATE)
            div_fac = 1 / np.max(np.abs(clean)) / 3.0
            clean = clean * div_fac

            pesq_array.append(pesq(SAMPLE_RATE, clean[:audio.shape[0]], audio))
            stoi_array.append(stoi(clean[:audio.shape[0]], audio, SAMPLE_RATE))

        pesq_array = list()
        stoi_array = list()

        print(f'STOI: {np.median(stoi_array)} PESQ: {np.median(pesq_array)}')
