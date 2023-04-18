import wandb
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Conv2D
from tensorflow.keras import Model
import numpy as np

tf.random.set_seed(999)
np.random.seed(999)

class SpeechModel:
    
    def __init__(self) -> None:
        self.window_length = 256
        self.sample_rate = 16000
        self.n_segments = 8
        self.n_clean_segments = 1
        self.n_fft = self.window_length
        self.n_features = self.n_fft//2 + 1
        self.overlap = round(0.25 * self.window_length)
        
        self._define_wandb_config()
        
    def _define_wandb_config(self):
        # Start a run, tracking hyperparameters
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project="speech_denoising",
            # track hyperparameters and run metadata with wandb.config
            config={
                "activation": "relu",
                "loss": "mse",
                "metric": "rmse",
                "optimizer": "adam",
                "epoch": 32,
                "batch_size" : 2048,
                "learning_rate" : 0.0015
            }
        )
        self.config = wandb.config
    
    def compile(self):
        self.model = self._build_model_architecture()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, epsilon = 1e-8)
        self.model.compile(optimizer=optimizer, loss=self.config.loss, metrics=[tf.keras.metrics.RootMeanSquaredError(self.config.metric)])
        
    def train(self, x_train, y_train, x_val, y_val):
        try:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode="min", restore_best_weights=True)
            history = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.config.epoch,
                batch_size = self.config.batch_size,
                validation_data=(x_val, y_val),
                callbacks=[
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint("models"),
                    early_stopping_callback
                ]
            )
            return history
        except AttributeError:
            print('There is no defined model now.')
        except:
            print('Unrecognized error. Make sure you pass the data for model training.')
    
    def show_architecture(self):
        try :
            print(self.model.summary())
        except:
            print('There is no defined model now.')
            
    def save(self):
        try:
            self.model.save('speech_model.h5')
            print('Model saved successfully!')
        except AttributeError:
            print("There is error with saving. Are you sure that this class has the trained model now?")
        
    def evaluate(self, x_test, y_test):
        try:
            val_loss = self.model.evaluate(x_test, y_test)[0]
            print(val_loss)
        except AttributeError:
            print("There is error with evaluation. Are you sure that this class has the trained model now?")
        
    # Because of skip connections, we need to use functional API
    def _build_model_architecture(self):
        m_activation = self.config.activation
        
        inputs = Input(shape=[self.n_features, self.n_segments, 1])
        model = inputs
        # -----
        model = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(model)
        model = Conv2D(filters=18, kernel_size=[9, 8], strides=[1, 1], padding='valid', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        skip0 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(skip0)
        model = BatchNormalization()(model)

        model = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        # -----
        model = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        skip1 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(skip1)
        model = BatchNormalization()(model)

        model = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        # ----
        model = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        # ----
        model = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = model + skip1
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        # ----
        model = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = model + skip0
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False)(model)
        model = Activation(m_activation)(model)
        model = BatchNormalization()(model)
        
        # ----
        model = tf.keras.layers.SpatialDropout2D(0.2)(model)
        model = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same')(model)

        model = Model(inputs=inputs, outputs=model)
        
        return model
    
    
