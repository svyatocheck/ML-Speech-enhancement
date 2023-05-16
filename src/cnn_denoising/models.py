import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
from src.cnn_denoising.config import *
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(999)
np.random.seed(999)


class SpeechModel:

    def __init__(self) -> None:
        self._define_wandb_config()
        self._compile(scratch=False)

    def _define_wandb_config(self):
        '''Инициализация платформы для прототипирования моделей.
        Содержит большинство гиперпараметров.'''
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project="speech_denoising",
            # track hyperparameters and run metadata with wandb.config
            config={
                "loss": "mse",
                "metric": "rmse",
                "optimizer": "adam",
                "epoch": 10,
                "batch_size": 64,
                "learning_rate": 0.001
            }, mode='online'
        )
        self.config = wandb.config

    def _compile(self, scratch=False):
        '''Создание модели.'''
        if scratch:
            self.model = self._create_model()
        else:
            try:
                self.model = tf.keras.models.load_model('src/cnn_denoising/models/speech_model.h5')
            except:
                print("No model in current directory. Aborting.")
                exit

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[
                           tf.keras.metrics.RootMeanSquaredError(self.config.metric)])

    def train(self, x_train, y_train, x_val, y_val):
        '''Запуск процесса обучения. На вход подаются вектора формата (?, 128, 8, 1)'''
        try:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=4, mode="min", restore_best_weights=True)

            learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
                self._step_decay)

            history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.config.epoch,
                batch_size=self.config.batch_size,
                validation_data=(x_val, y_val),
                callbacks=[
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint(
                        "models/speech_model.h5", save_best_only=True),
                    early_stopping_callback, learning_rate_scheduler
                ]
            )
            return history

        except AttributeError:
            print('There is no defined model now.')
        except:
            print('Unrecognized error. Make sure you pass the data for model training.')

    def show_architecture(self):
        '''Вывод архитектуры модели'''
        try:
            self.model.summary()
        except:
            print('There is no defined model now.')

    def save(self):
        '''Сохранение модели в директорию.'''
        try:
            self.model.save('models/speech_model.h5')
            print('Model saved successfully!')

        except AttributeError:
            print("There is error with saving. \
                    Are you sure that this class has the trained model now?")

    def evaluate(self, x_test, y_test):
        '''Запуск модели на тестовых данных.'''
        try:
            val_loss = self.model.evaluate(x_test, y_test)[0]
            print(val_loss)

        except AttributeError:
            print("There is error with evaluation. \
                    Are you sure that this class has the trained model now?")

    def _step_decay(self, epoch):
        '''Learning rate scheduler'''
        initial_lrate = self.config.learning_rate
        drop = 0.5
        epochs_drop = 5
        lrate = initial_lrate * \
            math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def _create_model(self):
        '''Создание архитектуры модели.'''
        inputs = tf.keras.layers.Input(shape=[N_FEATURES, N_SEGMENTS, 1])
        model = inputs
        # -----
        model = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(model)

        model = tf.keras.layers.Conv2D(filters=18, kernel_size=[9, 8], strides=[1, 1],
                                       padding='valid', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        skip0 = tf.keras.layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(skip0)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)

        # -----
        model = tf.keras.layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        skip1 = tf.keras.layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(skip1)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)

        # ----
        model = tf.keras.layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)

        # ----
        model = tf.keras.layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False,)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = model + skip1
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)

        # ----
        model = tf.keras.layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = model + skip0
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        # ---
        model = tf.keras.layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1],
                                       padding='same', use_bias=False)(model)
        model = tf.keras.layers.Activation(tf.nn.relu)(model)
        model = tf.keras.layers.BatchNormalization()(model)

        # ----
        model = tf.keras.layers.SpatialDropout2D(0.2)(model)
        model = tf.keras.layers.Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1],
                                       padding='same')(model)

        model = tf.keras.Model(inputs=inputs, outputs=model)

        return model
