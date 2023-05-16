DATA_PATH = '/run/media/svyatoslav/Files/Documents/speech/data'

CLEAN_TEST = f'{DATA_PATH}/input/clean_test/'

NOISY_TEST = f'{DATA_PATH}/input/noisy_test/'

SAMPLE_RATE = 16000

WINDOW_LENGTH = 256

OVERLAP = round(0.25 * WINDOW_LENGTH) # 50%

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1 # 257

N_SEGMENTS = 8