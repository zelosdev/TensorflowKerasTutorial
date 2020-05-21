############################
# turning off the warnings #

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#####################
# importing modules #

import numpy as np
import pickle
import matplotlib.pyplot as plt

from common.Viewer import *
from common.utils import create_directory

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

class WGAN():

    def __init__(self,
        input_dim, z_dim,
        d_conv_filters, d_conv_kernel_size, d_conv_strides, d_batch_norm_momentum, d_dropout_rate, # parameters for discriminator
        g_initial_dense_layer_size, g_upsample,                                                    # parameters for generator (upsampling)
        g_conv_filters, g_conv_kernel_size, g_conv_strides, g_batch_norm_momentum, g_dropout_rate, # parameters for generator (CNN)
    ):

        self.input_dim                  = input_dim
        self.z_dim                      = z_dim

        self.d_conv_filters             = d_conv_filters
        self.d_conv_kernel_size         = d_conv_kernel_size
        self.d_conv_strides             = d_conv_strides
        self.d_batch_norm_momentum      = d_batch_norm_momentum
        self.d_dropout_rate             = d_dropout_rate

        self.g_initial_dense_layer_size = g_initial_dense_layer_size
        self.g_upsample                 = g_upsample

        self.g_conv_filters             = g_conv_filters
        self.g_conv_kernel_size         = g_conv_kernel_size
        self.g_conv_strides             = g_conv_strides
        self.g_batch_norm_momentum      = g_batch_norm_momentum
        self.g_dropout_rate             = g_dropout_rate

        self._build()

    def _build(self):

        self._build_discriminator()
        self._build_generator()

        input_layer = Input(shape=self.z_dim, name='model_input')
        output_layer = self.discriminator(self.generator(input_layer))
        self.combined_model = Model(input_layer, output_layer)

    def _build_discriminator(self):

        num_layers = len(self.d_conv_filters)

        input_layer = Input(shape=self.input_dim, name='d_input')
        x = input_layer

        for i in range(num_layers):

            x = Conv2D(
                filters     = self.d_conv_filters[i],
                kernel_size = self.d_conv_kernel_size[i],
                strides     = self.d_conv_strides[i],
                padding     = 'same',
                name        = 'd_conv_' + str(i)
            )(x)

            if self.d_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.d_batch_norm_momentum)(x)

            x = LeakyReLU()(x)

            if self.d_dropout_rate:
                x = Dropout(rate=self.d_dropout_rate)(x)

        x = Flatten()(x)

        # Difference 1 compared to vanilla GAN
        output_layer = Dense(1, activation=None)(x)

        self.discriminator = Model(input_layer, output_layer)

    def _wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def _build_generator(self):

        num_layers = len(self.g_conv_filters)

        input_layer = Input(shape=self.z_dim, name='g_input')
        x = input_layer

        x = Dense(np.prod(self.g_initial_dense_layer_size))(x)

        if self.g_batch_norm_momentum:
            x = BatchNormalization(momentum=self.g_batch_norm_momentum)(x)

        x = LeakyReLU()(x)

        x = Reshape(self.g_initial_dense_layer_size)(x)

        if self.g_dropout_rate:
            x = Dropout(rate=self.g_dropout_rate)(x)

        for i in range(num_layers):

            if self.g_upsample[i] == 2:

                x = UpSampling2D()(x)
                x = Conv2D(
                    filters     = self.g_conv_filters[i],
                    kernel_size = self.g_conv_kernel_size[i],
                    padding     = 'same',
                    name        = 'g_conv_' + str(i)
                )(x)

            else:

                x = Conv2DTranspose(
                    filters     = self.g_conv_filters[i],
                    kernel_size = self.g_conv_kernel_size[i],
                    padding     = 'same',
                    strides     = self.g_conv_strides[i],
                    name        = 'g_conv_' + str(i)
                )(x)

            if i < num_layers - 1:

                if self.g_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.g_batch_norm_momentum)(x)

                x = LeakyReLU()(x)

            else:

                x = Activation('tanh')(x)

        output_layer = x

        self.generator = Model(input_layer, output_layer)

    def compile(self, d_learning_rate, g_learning_rate):

        self.discriminator.compile(loss=self._wasserstein, optimizer=RMSprop(learning_rate=d_learning_rate))
        self.combined_model.compile(loss=self._wasserstein, optimizer=RMSprop(learning_rate=g_learning_rate))

    def _train_D(self, x_train, batch_size, clip_threshold):

        # WARNING:tensorflow:Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?
        self.discriminator.trainable = True

        # Difference 2 compared to vanilla GAN
        real = np.ones((batch_size, 1)) # the answer for the decision as real
        fake = -np.ones((batch_size,1)) # thwe answer for the decision as fake

        num_training_images = x_train.shape[0] # x_train.shape = (80000, 28, 28, 1)
        idx = np.random.randint(0, num_training_images, batch_size)
        true_imgs = x_train[idx] # how many images?: batch_size

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise) # how many images?: batch_size

        loss_real = self.discriminator.train_on_batch(true_imgs, real)
        loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        loss_D = 0.5 * (loss_real + loss_fake)

        # Difference 3 compared to vanilla GAN
        for layer in self.discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            layer.set_weights(weights)

        return [loss_D, loss_real, loss_fake]

    def _train_G(self, batch_size):

        # WARNING:tensorflow:Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?
        self.discriminator.trainable = False

        real = np.ones((batch_size, 1)) # the answer for the decision as real
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        loss_G = self.combined_model.train_on_batch(noise, real)

        return loss_G

    def train(self, x_train, batch_size, epochs, num_critics, clip_threshold):

        for epoch in range(epochs):

            # Difference 4 compared to vanilla GAN
            for _ in range(num_critics):
                loss_D = self._train_D(x_train, batch_size, clip_threshold)

            loss_G = self._train_G(batch_size)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, loss_D[0], loss_D[1], loss_D[2], loss_G))

    def save(self, export_path):

        create_directory(export_path)

        # The number and order of constructor arguments must match.
        with open(os.path.join(export_path, 'params.pkl'), 'wb') as f:
            pickle.dump([ # the arguments of the class constructor
                self.input_dim, self.z_dim,
                self.d_conv_filters, self.d_conv_kernel_size, self.d_conv_strides, self.d_batch_norm_momentum, self.d_dropout_rate,
                self. g_initial_dense_layer_size, self.g_upsample,
                self.g_conv_filters, self.g_conv_kernel_size, self.g_conv_strides, self.g_batch_norm_momentum, self.g_dropout_rate,
            ], f)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)