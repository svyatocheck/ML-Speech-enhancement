from speech_model import SpeechModel
import pickle

DATA_PATH = '/run/media/svyatoslav/Files/ml_data_project/'

def read_test_data():
    x_test = pickle.load(open(DATA_PATH + 'X_test_1.pkl', 'rb'))
    y_test = pickle.load(open(DATA_PATH + 'Y_test_1.pkl', 'rb'))
    return x_test, y_test

def read_train_data():
    x_train = pickle.load(open(DATA_PATH + 'X_training_3.pkl', 'rb'))
    y_train = pickle.load(open(DATA_PATH + 'Y_training_3.pkl', 'rb'))
    return x_train, y_train

def read_validation_data():
    x_val = pickle.load(open(DATA_PATH + 'X_val_1.pkl', 'rb'))
    y_val = pickle.load(open(DATA_PATH + 'Y_val_1.pkl', 'rb'))
    return x_val, y_val


def main():
    model = SpeechModel()
    model.compile()
    
    x_train, y_train = read_train_data()
    x_val, y_val = read_validation_data()
    
    model.train(x_train, y_train, x_val, y_val)
    
    del x_train, y_train
    del x_val, y_val
    
    x_test, y_test = read_test_data()
    model.evaluate(x_test, y_test)
    
    model.save()
    

if __name__ == '__main__':
    main()