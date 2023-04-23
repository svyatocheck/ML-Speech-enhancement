import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import numpy as np
from constants import *

import logging
tf.get_logger().setLevel(logging.ERROR)

tf.random.set_seed(999)
np.random.seed(999)

class SpeechModel:

    def __init__(self) -> None:
        self._define_wandb_config()
        self._compile()

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
                "batch_size": 32,
                "learning_rate": 0.002
            }, mode='offline'
        )
        self.config = wandb.config

    def train(self, x_train, y_train, x_val, y_val):
        '''Запуск процесса обучения. На вход подаются вектора формата (?, 257, 8, 1)'''
        try:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=4, mode="min", restore_best_weights=True)

            history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.config.epoch,
                batch_size=self.config.batch_size,
                validation_data=(x_val, y_val),
                callbacks=[
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint(
                        "models/test.h5", save_best_only=True),
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

    def _compile(self, scratch=True):
        '''Создание модели.'''
        if scratch:
            self.model = self._create_model()
        else:
            # Доработаю этот момент позже.
            self.model = tf.keras.models.load_model('model.h5')

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[
                           tf.keras.metrics.MeanAbsoluteError(self.config.metric)])

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
