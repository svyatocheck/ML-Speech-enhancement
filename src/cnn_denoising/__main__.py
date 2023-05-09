from src.cnn_denoising.configs import *
from src.cnn_denoising.models import SpeechModel
from src.cnn_denoising.features import FeatureInputGenerator
from glob import glob
import random
import math

''' Launch cnn model training. Before start, make sure, that all wandb config are good for you. Or just change code to remove it. '''

def prepare_data(generator : FeatureInputGenerator, files_type):
    x_data, y_data = generator.start_preprocess(files_type)
    return x_data, y_data

def execute_main():
    # train_size = int(input("Train dataset size: "))
    train_size = 512
    validation_size = math.floor((train_size * 10) / 100)
    test_size = validation_size

    train_files = glob(f'{DATA_PATH}/input/noisy_train/*.wav')
    test_files = glob(f'{DATA_PATH}/input/noisy_test/*.wav')

    random.shuffle(train_files)
    random.shuffle(test_files)

    train_files = train_files[:train_size + validation_size]
    test_files = test_files[:test_size]
    validation_files = train_files[:validation_size]
    train_files = train_files[validation_size:]

    generator = FeatureInputGenerator()

    x_val, y_val = prepare_data(generator, validation_files)
    x_train, y_train = prepare_data(generator, train_files)
    
    model = SpeechModel()
    model.train(x_train, y_train, x_val, y_val)
    
    x_test, y_test = prepare_data(generator, test_files)
    model.evaluate(x_test, y_test)
    
    # model.save() # WanDB saves model automatically


if __name__ == '__main__':
    execute_main()