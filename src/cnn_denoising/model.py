import math
import os

import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from src.cnn_denoising.config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(999)
np.random.seed(999)


class CNN_Model:

    def __init__(self) -> None:
        self._define_wandb_config()
        self._compile(scratch=False)


    def _define_wandb_config(self):
        """
        Init wandb for tracking hyperparameters and run metadata
        :return: None
        """
        # define wandb config
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
            }, mode='offline'
        )
        self.config = wandb.config


    def _compile(self, scratch=False):
        """
        Compile model instance.
        :param scratch: create new or use existing model
        :param path_to_previous: path to already existing model
        :return: tensorflow model
        """
        if scratch:
            self.model = self._create_model()
        else:
            try:
                self.model = tf.keras.models.load_model(
                    'src/cnn_denoising/models/speech_model.h5')
            except:
                print("No model in current directory. Aborting.")
                exit

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[
                           tf.keras.metrics.RootMeanSquaredError(self.config.metric)])


    def train(self, x_train, y_train, x_val, y_val):
        """
        Feed model with prepared features.
        :param x_train, x_val: consecutive noisy STFT magnitude vectors (?, 128, 8, 1)
        :param y_train, y_val: consecutive clean STFT magnitude vectors
        :return: history
        """
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
        """
        Print model summary
        :return: None
        """
        try:
            self.model.summary()
        except:
            print('There is no defined model now.')


    def save(self):
        """
        Save model locally. Keep in mind, that wandb does it automatically.
        :param path_to_model: path to model
        :return: None
        """
        try:
            self.model.save('models/speech_model.h5')
            print('Model saved successfully!')

        except AttributeError:
            print("There is error with saving. \
                    Are you sure that this class has the trained model now?")


    def evaluate(self, x_test, y_test):
        """
        Launch model on test data.
        :param x_test: consecutive noisy STFT magnitude vectors (?, 128, 8, 1)
        :param y_train: consecutive clean STFT magnitude vectors (?, 128, 8, 1)
        :return: None
        """
        try:
            val_loss = self.model.evaluate(x_test, y_test)[0]
            print(val_loss)

        except AttributeError:
            print("There is error with evaluation. \
                    Are you sure that this class has the trained model now?")


    def _step_decay(self, epoch):
        """
        Function to schedule learning rate. Needs to be added to callbacks
        :param epoch: Number of current epoch 
        :return: new learning rate
        """
        initial_lrate = self.config.learning_rate
        drop = 0.5
        epochs_drop = 5
        lrate = initial_lrate * \
            math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def _create_model(self):
        """
        Architecture of the model. Main function.
        :return: tensorflow model instance.
        """
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
