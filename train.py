from constants import *
from speech_model import SpeechModel
from audio_generator import FeatureInputGenerator
from signal import signal, SIGPIPE, SIG_DFL
import random
from glob import glob
import math

signal(SIGPIPE,SIG_DFL)

TRAIN_SIZE = 2048
VAL_SIZE = math.floor((TRAIN_SIZE * 10) / 100) # 10%
TEST_SIZE = VAL_SIZE

NOIZY_FILES = glob(f'{DATA_PATH}/noisy/*.wav')
TEST_FILES = glob(f'{DATA_PATH}/noisy_test/*.wav')

random.shuffle(NOIZY_FILES)
random.shuffle(TEST_FILES)

NOIZY_FILES = NOIZY_FILES[:TRAIN_SIZE + VAL_SIZE]
TEST_FILES = TEST_FILES[:TEST_SIZE]
VALIDATION_FILES = NOIZY_FILES[:VAL_SIZE]
NOIZY_FILES = NOIZY_FILES[VAL_SIZE:]


def prepare_data(generator : FeatureInputGenerator, files_type):
    x_data, y_data = generator.start_preprocess(files_type)
    return x_data, y_data


def main():
    model = SpeechModel()
    generator = FeatureInputGenerator()
    
    x_val, y_val = prepare_data(generator, VALIDATION_FILES)
    x_train, y_train = prepare_data(generator, NOIZY_FILES)
    
    model.train(x_train, y_train, x_val, y_val)
    
    x_test, y_test = prepare_data(generator, TEST_FILES)
    model.evaluate(x_test, y_test)
    
    model.save()


if __name__ == '__main__':
    main()