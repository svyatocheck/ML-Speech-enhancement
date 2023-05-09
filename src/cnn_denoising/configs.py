
DATA_PATH = '/run/media/svyatoslav/Files/Documents/speech/data'

SPEECH_DATASET = f'{DATA_PATH}/input/speech/'

CLEAN_TRAIN = f'{DATA_PATH}/input/clean_train/'

CLEAN_TEST = f'{DATA_PATH}/input/clean_test/'

SAMPLE_RATE = 16000

WINDOW_LENGTH = 256

OVERLAP = round(0.25 * WINDOW_LENGTH)

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1 

N_SEGMENTS = 8