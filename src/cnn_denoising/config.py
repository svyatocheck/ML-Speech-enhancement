SAMPLE_RATE = 16000

WINDOW_LENGTH = 256

OVERLAP = round(0.25 * WINDOW_LENGTH)

N_FFT = WINDOW_LENGTH

N_FEATURES = N_FFT // 2 + 1 

N_SEGMENTS = 8