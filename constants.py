
DATA_PATH = '/run/media/svyatoslav/Files/Documents/speech/data'

DATASET_FIRST = f'{DATA_PATH}/input/speech/'

DATASET_SECOND = f'{DATA_PATH}/input/speech_clean_train/'

TEST_DATASET = f'{DATA_PATH}/input/speech_clean_test/'

SAMPLE_RATE = 16000

WINDOW_LENGTH = 256

OVERLAP = round(0.25 * WINDOW_LENGTH) # 50%

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1 # 257

N_SEGMENTS = 8