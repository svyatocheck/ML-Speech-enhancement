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
        """
        Init wandb for tracking hyperparameters and run metadata
        :return: None
        """
        # define wandb config
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
        """
        Feed model with prepared features.
        :param x_train, x_val: consecutive noisy STFT magnitude vectors (?, 257, 1, 1)
        :param y_train, y_val: consecutive clean STFT magnitude vectors
        :return: history
        """
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
                        "src/crnn_denoising/models/speech_model.h5", save_best_only=True),
                    early_stopping_callback
                ]
            )
            return history

        except AttributeError:
            print('There is no defined model now.')
        except:
            print('Unrecognized error. Make sure you pass the data for model training.')


    def show_architecture(self):
        """
        Print model summary
        :return: None
        """
        try:
            self.model.summary()
        except:
            print('There is no defined model now.')


    def save(self, path_to_model):
        """
        Save model locally. Keep in mind, that wandb does it automatically.
        :param path_to_model: path to model
        :return: None
        """
        try:
            self.model.save(path_to_model)
            print('Model saved successfully!')
        except AttributeError:
            print(
                "There is error with saving. Are you sure that this class has the trained model now?")


    def evaluate(self, x_test, y_test):
        """
        Launch model on test data.
        :param x_test: consecutive noisy STFT magnitude vectors (?, 257, 1, 1)
        :param y_train: consecutive clean STFT magnitude vectors (?, 257, 1, 1)
        :return: None
        """
        try:
            val_loss = self.model.evaluate(x_test, y_test)[0]
            print(val_loss)
        except AttributeError:
            print(
                "There is error with evaluation. Are you sure that this class has the trained model now?")


    def _step_decay(self, epoch):
        """
        Function to schedule learning rate. Needs to be added to callbacks
        :param epoch: Number of current epoch 
        :return: new learning rate
        """
        initial_lrate = self.config.learning_rate
        drop = 0.75
        epochs_drop = 5.0
        lrate = initial_lrate * \
            math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


    def _compile(self, scratch=False, path_to_previous=""):
        """
        Compile model instance.
        :param scratch: create new or use existing model
        :param path_to_previous: path to already existing model
        :return: tensorflow model
        """
        if scratch:
            self.model = self._create_model()
        else:
            # Доработаю этот момент позже.
            self.model = tf.keras.models.load_model(path_to_previous)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[
                           tf.keras.metrics.MeanAbsoluteError(self.config.metric)])


    def _create_model(self):
        """
        Architecture of the model. Main function.
        :return: tensorflow model instance.
        """
        input_first = tf.keras.layers.Input(shape=[N_FEATURES, N_SEGMENTS, 1])
        # ---
        crnn_first = self._make_first_part(input_first)
        crnn_second = self._make_first_part(input_first)
        # ---
        concatenated = tf.keras.layers.Add()([crnn_first, crnn_second])

        # Fully connected layers
        dense_1 = tf.keras.layers.Dense(
            257, activation=tf.nn.relu, use_bias=True)(concatenated)
        dense_2 = tf.keras.layers.Dense(257, use_bias=True)(dense_1)

        # ---
        reshaped = tf.keras.layers.Reshape((257, 1, 1))(dense_2)
        model = tf.keras.Model(inputs=input_first, outputs=reshaped)
        return model


    def _make_first_part(self, inputs):
        """
        Architecture of the model. Help function.
        :return: tensorflow layers.
        """
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

        # ---
        x = tf.reshape(conv4, [-1, conv4.shape[1],
                       conv4.shape[2] * conv4.shape[3]])

        lstm_fw_cell_1 = tf.keras.layers.LSTM(
            33, return_sequences=True, activation=tf.nn.relu)(x)
        drop_one = tf.keras.layers.Dropout(0.1)(lstm_fw_cell_1)

        lstm_fw_cell_2 = tf.keras.layers.LSTM(
            33, return_sequences=True, activation=tf.nn.relu)(drop_one)
        drop_two = tf.keras.layers.Dropout(0.1)(lstm_fw_cell_2)
        # ---
        flatten = tf.keras.layers.Flatten()(drop_two)
        return flatten
