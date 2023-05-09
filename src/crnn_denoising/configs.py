DATA_PATH = '/run/media/svyatoslav/Files/Documents/speech/data'

SPEECH_DATASET = f'{DATA_PATH}/input/speech/'

CLEAN_TRAIN = f'{DATA_PATH}/input/clean_train/'

CLEAN_TEST = f'{DATA_PATH}/input/clean_test/'

NOISE_TRAIN = f'{DATA_PATH}/input/noise/'

NOISE_TEST = f'{DATA_PATH}/input/noise_test/'

SAMPLE_RATE = 16000

WINDOW_LENGTH = 512

OVERLAP = round(0.5 * WINDOW_LENGTH) # 50%

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1 # 257

N_SEGMENTS = 1