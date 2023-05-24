import wandb
from src.crnn_denoising.config import *
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import numpy as np
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(999)
np.random.seed(999)

class SpeechModel:

    def __init__(self) -> None:
        self._define_wandb_config()
        self._compile()

    def _define_wandb_config(self):
        '''Инициализация платформы для прототипирования моделей.
        Содержит большинство гиперпараметров.'''
        
        # track hyperparameters and run metadata with wandb.config
        self.run = wandb.init(
            project="crnn_denoising",
            config={
                "loss": "mse",
                "metric": "mae",
                "optimizer": "adam",
                "epoch": 50,
                "batch_size": 64,
                "learning_rate": 0.001
            }, mode='offline'
        )
        self.config = wandb.config


    def train(self, x_train, y_train, x_val, y_val):
        '''Запуск процесса обучения. На вход подаются вектора формата (?, 257, 1, 1)'''
        try:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode="min", restore_best_weights=True)
            history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.config.epoch,
                batch_size=self.config.batch_size,
                validation_data=(x_val, y_val),
                callbacks=[
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint("src/crnn_denoising/models/speech_model.h5", save_best_only=True),
                    early_stopping_callback
                ]
            )
            return history
        
        except AttributeError:
            print('There is no defined model now.')
        except:
            print('Unrecognized error. Make sure you pass the data for model training.')

    def show_architecture(self):
        '''Вывод архитектуры модели, ничего особенного. 
        plot_model не смог завестись.'''
        try:
            self.model.summary()
        except:
            print('There is no defined model now.')

    def save(self):
        '''Сохранение модели на платформу для прототипирования и в текущую директорию.'''
        try:
            self.model.save('speech_model.h5')
            print('Model saved successfully!')
        except AttributeError:
            print(
                "There is error with saving. Are you sure that this class has the trained model now?")

    def evaluate(self, x_test, y_test):
        '''Запуск модели на тестовых данных.'''
        try:
            val_loss = self.model.evaluate(x_test, y_test)[0]
            print(val_loss)
        except AttributeError:
            print(
                "There is error with evaluation. Are you sure that this class has the trained model now?")
    
    # # learning rate schedule
    # def _step_decay(self, epoch):
    #     initial_lrate = self.config.learning_rate
    #     drop = 0.75
    #     epochs_drop = 5.0
    #     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #     return lrate

    def _compile(self, scratch = True):
        '''Создание модели.'''
        if scratch:
            self.model = self._create_model()
        else:
            # Доработаю этот момент позже.
            self.model = tf.keras.models.load_model('src/crnn_denoising/models/speech_model.h5')
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[
                           tf.keras.metrics.MeanAbsoluteError(self.config.metric)])

    def _create_model(self):
        '''Создание архитектуры модели.'''
        input_first = tf.keras.layers.Input(shape=[N_FEATURES, N_SEGMENTS, 1])
        # ---
        crnn_first = self._make_first_part(input_first)
        crnn_second = self._make_first_part(input_first)
        # ---
        concatenated = tf.keras.layers.Add()([crnn_first, crnn_second])

        # Fully connected layers
        dense_1 = tf.keras.layers.Dense(257, activation=tf.nn.relu, use_bias=True)(concatenated)
        dense_2 = tf.keras.layers.Dense(257, use_bias=True)(dense_1)
        
        # ---
        reshaped = tf.keras.layers.Reshape((257, 1, 1))(dense_2)
        model = tf.keras.Model(inputs=input_first, outputs=reshaped)
        return model


    def _make_first_part(self, inputs):
        # ---
        conv1 = tf.keras.layers.Conv2D(filters=257, kernel_size=(5, 5),
                                       padding="same", activation=tf.nn.relu)(inputs)
        pool1 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=2, padding="same")(conv1)

        # ---
        conv2 = tf.keras.layers.Conv2D(filters=129, kernel_size=(5, 5),
                                       padding="same", activation=tf.nn.relu)(pool1)
        pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=2, padding="same")(conv2)

        # ---
        conv3 = tf.keras.layers.Conv2D(filters=65, kernel_size=(5, 5),
                                       padding="same", activation=tf.nn.relu)(pool2)
        pool3 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=2, padding="same")(conv3)

        # ---
        conv4 = tf.keras.layers.Conv2D(filters=33, kernel_size=(5, 5),
                                       padding="same", activation=tf.nn.relu)(pool3)

        x = tf.reshape(conv4, [-1, conv4.shape[1],
                       conv4.shape[2] * conv4.shape[3]])

        lstm_fw_cell_1 = tf.keras.layers.LSTM(33, return_sequences=True, activation=tf.nn.relu)(x)
        drop_one = tf.keras.layers.Dropout(0.1)(lstm_fw_cell_1)
        lstm_fw_cell_2 = tf.keras.layers.LSTM(33, return_sequences=True, activation=tf.nn.relu)(drop_one)
        drop_two = tf.keras.layers.Dropout(0.1)(lstm_fw_cell_2)
        # ---
        flatten = tf.keras.layers.Flatten()(drop_two)
        return flatten
