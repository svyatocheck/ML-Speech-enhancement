import math
import random
from glob import glob

import src.cnn_denoising.config as cnn_config
import src.crnn_denoising.config as crnn_config
from src.cnn_denoising.model import CNN_Model
from src.config import *
from src.crnn_denoising.model import CRNN_Model
from src.dataset_creation.dataset_creator import Dataset_Creator
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.model_evaluate.evaluator import ModelEvaluator

''' Launch crnn model training. Before start, make sure, that all wandb config are good for you. Or just change code to remove it. '''

# Define sizes of parts in dataset for curent experiment
train_size = 10
validation_size = math.floor((train_size * 10) / 100)
test_size = math.floor((train_size * 20) / 100)


def prepare_data(generator: FeatureExtractor, files_type):
    x_data, y_data = generator.start_preprocess(files_type)
    return x_data, y_data


def train_model(is_crnn: bool):
    train_files = glob(f'{RESULT_DATASET_TRAIN}*')
    random.shuffle(train_files)

    # Cut train dataset list to defined size
    train_files = train_files[:train_size + validation_size]
    validation_files = train_files[:validation_size]
    train_files = train_files[validation_size:]

    # Extract features from those audios
    if is_crnn:
        generator = FeatureExtractor(crnn_config.SAMPLE_RATE, crnn_config.WINDOW_LENGTH, crnn_config.N_FFT,
                                     crnn_config.OVERLAP, crnn_config.N_FEATURES, crnn_config.N_SEGMENTS, is_crnn)
    else:
        generator = FeatureExtractor(cnn_config.SAMPLE_RATE, cnn_config.WINDOW_LENGTH, cnn_config.N_FFT,
                                     cnn_config.OVERLAP, cnn_config.N_FEATURES, cnn_config.N_SEGMENTS, is_crnn)

    x_val, y_val = prepare_data(generator, validation_files)
    x_train, y_train = prepare_data(generator, train_files)

    model = CRNN_Model() if is_crnn else CNN_Model()
    model.train(x_train, y_train, x_val, y_val)

    # Cut test dataset list to defined size
    test_files = glob(f'{RESULT_DATASET_TEST}*')
    random.shuffle(test_files)
    test_files = test_files[:test_size]

    # Test model
    x_test, y_test = prepare_data(generator, test_files)
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    choise = input(
        "What do you want to do?\n1 - Train CNN model, 2 - Train CRNN model, 3 - Prepare noisy dataset, 4 - Evaluate model\n")
    print("")
    if choise == '1':
        train_model(False)
    elif choise == '2':
        train_model(True)
    elif choise == '3':
        Dataset_Creator().execute_speech_preparation()
    elif choise == '4':
        ModelEvaluator(glob(f'{RESULT_DATASET_TRAIN}*')).evaluate()
    else:
        exit
